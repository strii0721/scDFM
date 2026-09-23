#!/usr/bin/env python3
"""守护分发进程：每 poll_s(10) 秒轮询所有 GPU 显存，空闲（可用 > free_mb=50G）的
卡立刻领下一个 panel 基因的推理任务。每基因一个独立进程：跑完立即写
predparts/pred_shard_disp_{task}_g000.h5ad 并退出（释放显存），下个轮询周期该卡
若仍空闲则继续领任务；全部 300 基因完成后自动 eval_only 出最终六指标
（scores.csv 落在 out_dir）。

与旧分片机制的区别：单基因粒度调度 + 显存门控——别人占用某卡（可用 <50G）时
该卡自动被跳过，别人腾出后自动恢复；进程级失败自动重试（--max_retry）。

断电/重启安全：启动时扫描 predparts 里全部基因级 part（含旧分片命名），已完成
基因自动跳过；dispatch_map.txt 记录 disp 任务号→基因映射（编号续接）。

用法（在项目根目录）:
  cd /home/ict2/Projects/scDFM
  .venv/bin/python -m src.script.bench_dispatch \
      --checkpoint_path output/train/2026-09-22_23-10/iteration_21000/checkpoint.pt \
      --out_dir output/benchmark/replogle_rpe1 --gpus 7 --ode_steps 12
"""
import argparse
import glob
import os
import subprocess
import sys
import time

import anndata as ad


def free_mem_mib() -> dict[int, int]:
    """{gpu_idx: free_mib}，解析 nvidia-smi 的 total/used（MiB）。"""
    out = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.total,memory.used',
         '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True).stdout
    res = {}
    for line in out.strip().splitlines():
        idx, tot, used = (int(x.strip()) for x in line.split(','))
        res[idx] = tot - used
    return res


def load_done(out_dir: str) -> set[str]:
    """扫描 predparts 全部基因级 part（含旧分片命名），解析已完成基因名集合。"""
    done = set()
    for p in glob.glob(os.path.join(out_dir, 'predparts', '*_g*.h5ad')):
        try:
            a = ad.read_h5ad(p)
            done.add(str(a.obs['target_gene'].iloc[0]))
        except Exception as e:
            print(f'[warn] 跳过坏 part {os.path.basename(p)}: {e}', flush=True)
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint_path', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--heldout_line', default='RPE1')
    ap.add_argument('--gpus', type=int, default=7)
    ap.add_argument('--ode_steps', type=int, default=12)
    ap.add_argument('--batch_size', type=int, default=5)
    ap.add_argument('--free_mb', type=int, default=51200, help='空闲显存门控（默认 50G）')
    ap.add_argument('--poll_s', type=float, default=10.0, help='轮询周期（秒）')
    ap.add_argument('--max_retry', type=int, default=3)
    ap.add_argument('--settle_s', type=float, default=0.0,
                    help='启动后先等待 N 秒再扫描 done/下发（供重启时等在飞旧 worker 跑完）')
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(root)
    out_dir = os.path.abspath(args.out_dir)

    # 单实例锁
    lock = os.path.join(out_dir, '.dispatch.pid')
    if os.path.exists(lock):
        try:
            pid = int(open(lock).read().strip())
            os.kill(pid, 0)
            print(f'[abort] 已有 dispatcher 在跑 (pid={pid}, {lock})', flush=True)
            sys.exit(1)
        except (ValueError, ProcessLookupError):
            pass
    open(lock, 'w').write(str(os.getpid()))

    if args.settle_s > 0:
        print(f'[dispatch] 等待在飞 worker 完成 {args.settle_s:.0f}s 后再扫描', flush=True)
        time.sleep(args.settle_s)

    log_dir = 'logs/dispatch'
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(out_dir, 'perts.txt')) as f:
        all_genes = [l.strip() for l in f if l.strip()]
    done = load_done(out_dir)
    queue = [g for g in all_genes if g not in done]
    print(f'[dispatch] {len(queue)} 待推理 / {len(done)} 已完成（共 {len(all_genes)}）', flush=True)

    map_path = os.path.join(out_dir, 'dispatch_map.txt')
    task_next = 0
    if os.path.exists(map_path):
        with open(map_path) as f:
            task_next = sum(1 for _ in f)
    map_f = open(map_path, 'a')

    procs: dict[int, tuple[int, str, subprocess.Popen]] = {}  # gpu -> (task, gene, proc)
    retry: dict[str, int] = {}
    gave_up: list[str] = []

    def spawn(g: int, gene: str) -> None:
        nonlocal task_next
        task = task_next
        task_next += 1
        map_f.write(f'{task}\t{gene}\n')
        map_f.flush()
        log = os.path.join(log_dir, f'g{g:02d}_t{task:03d}_{gene}.log')
        cmd = [
            os.path.join(root, '.venv/bin/python'), '-m',
            'src.script.benchmark_line_holdout',
            '--checkpoint_path', args.checkpoint_path,
            '--heldout_line', args.heldout_line,
            '--out_dir', out_dir,
            '--no_eval', '--reuse_real',
            '--batch_size', str(args.batch_size),
            '--ode_steps', str(args.ode_steps),
            '--seed', str(42 + task),
            '--perts', gene,
            '--pred_tag', f'shard_disp_{task}',
            '--parts_only',
        ]
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(g)
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        proc = subprocess.Popen(cmd, env=env, stdout=open(log, 'w'),
                                stderr=subprocess.STDOUT,
                                start_new_session=True)  # 脱离 dispatcher 进程组：
        # dispatcher/tmux 被杀不连坐 worker，在飞基因跑完照常落盘
        procs[g] = (task, gene, proc)
        print(f'[dispatch] gpu{g} <- t{task} {gene} '
              f'({time.strftime("%H:%M:%S")})', flush=True)

    while queue or procs:
        free = free_mem_mib()
        # 收割已退出 worker
        for g, (task, gene, proc) in list(procs.items()):
            if proc.poll() is not None:
                part = os.path.join(out_dir, 'predparts',
                                    f'pred_shard_disp_{task}_g000.h5ad')
                ok = os.path.exists(part)
                del procs[g]
                if ok:
                    done.add(gene)
                    print(f'[dispatch] gpu{g} t{task} {gene} 完成 '
                          f'({len(done)}/{len(done) + len(queue)})', flush=True)
                else:
                    retry[gene] = retry.get(gene, 0) + 1
                    if retry[gene] > args.max_retry:
                        gave_up.append(gene)
                        print(f'[ERROR] {gene} 失败 {retry[gene]} 次，放弃 '
                              f'（日志 logs/dispatch/g{g:02d}_t{task:03d}_{gene}.log）', flush=True)
                    else:
                        queue.append(gene)
                        print(f'[retry] {gene} 失败（第 {retry[gene]} 次），重新入队', flush=True)
        # 空闲（可用 > 门控）且无我方进程的卡领任务
        for g in range(args.gpus):
            if g in procs:
                continue
            if free.get(g, 0) > args.free_mb and queue:
                spawn(g, queue.pop(0))
        time.sleep(args.poll_s)

    map_f.close()
    os.remove(lock)
    if gave_up:
        print(f'[dispatch] 失败放弃 {len(gave_up)}: {gave_up}', flush=True)
    print(f'[dispatch] 全部 {len(done)} 基因完成，启动最终 eval_only', flush=True)
    subprocess.run([
        os.path.join(root, '.venv/bin/python'), '-m',
        'src.script.benchmark_line_holdout',
        '--checkpoint_path', args.checkpoint_path,
        '--heldout_line', args.heldout_line,
        '--out_dir', out_dir,
        '--eval_only',
    ], check=True)
    print(f'[dispatch] 完成：最终 scores.csv = {os.path.join(out_dir, "scores.csv")}',
          flush=True)


if __name__ == '__main__':
    main()
