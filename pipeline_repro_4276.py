"""mlx#4276: try to provoke EFAULT on a ring send/recv round trip.

    mlx.launch --backend ring --hostfile two_ranks.json pipeline_repro.py
    mlx.launch ... pipeline_repro.py --rounds 300 --free-early --ballast-gb 20

The report is multi-node pipeline inference over the ring backend, where the
receiving rank fails with errno 14 on recv, repeats to the ten error threshold
and aborts.

**This is not a pipeline.** Rank 0 sends a tensor, rank 1 receives it and sends
the same tensor back. There is no model, no stage to stage forward flow and no
KV cache, so a fault that needs the traffic pattern of a real pipeline will not
show up here. What it does is exercise the send/recv path itself.

EFAULT means the pointer handed to recv(2) was not valid, so everything here
aims at the receive buffer rather than at moving bytes quickly:

  1. buffers are produced on the GPU rather than `mx.ones`, so the array handed
     to send is the output of a Metal kernel;
  2. the receive target is freshly allocated every round, so the allocator is
     recycling underneath the transport;
  3. sizes alternate and are deliberately not round, so the segment arithmetic
     never divides evenly;
  4. --free-early drops every reference and clears the cache the instant the
     collective returns;
  5. --ballast-gb holds a large resident allocation to stand in for a shard.

Result on 2x Mac16,11 M4 Pro, mlx 0.32.0, over both a LAN and a direct
Thunderbolt link: 500 rounds plus 200 under 20 GB of ballast, no EFAULT.
"""

import argparse
import sys

import mlx.core as mx
import mlx.core.distributed as dist


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=200)
    p.add_argument("--ballast-gb", type=float, default=0.0,
                   help="hold this many GB resident, like a model shard")
    p.add_argument("--free-early", action="store_true",
                   help="drop the python reference right after the collective")
    args = p.parse_args()

    g = dist.init()
    r, n = g.rank(), g.size()
    if n != 2:
        raise SystemExit("this repro wants exactly 2 ranks")
    peer = 1 - r

    def mark(*a):
        print(f"[rank {r}]", *a, file=sys.stderr, flush=True)

    ballast = None
    if args.ballast_gb > 0:
        # Stand in for a model shard: a large resident allocation that keeps
        # the allocator and the residency set under pressure while the
        # transport is writing into freshly allocated receive buffers.
        # Two dimensional because a single dimension over 2**31 elements
        # overflows the shape type long before it overflows the memory.
        cols = 1 << 16
        rows = int(args.ballast_gb * (1 << 30) // 4) // cols
        ballast = mx.random.normal((rows, cols)).astype(mx.float32)
        mx.eval(ballast)
        mark(f"holding {args.ballast_gb:g} GB resident")

    mark("init done, starting pipeline rounds")
    # Decode-like and prefill-like, and deliberately not round numbers so the
    # segment arithmetic never divides evenly.
    sizes = [4096, 1_000_003, 16_777_213, 67_108_859]

    for i in range(args.rounds):
        elems = sizes[i % len(sizes)]
        # Produced by a GPU kernel, the way an activation is, rather than a
        # constant the allocator may treat differently.
        x = (mx.random.normal((elems,)) * 2.0).astype(mx.float32)
        mx.eval(x)

        if r == 0:
            mx.eval(dist.send(x, peer, group=g))
            back = dist.recv_like(x, peer, group=g)
            mx.eval(back)
        else:
            got = dist.recv_like(x, peer, group=g)
            mx.eval(got)
            mx.eval(dist.send(got, peer, group=g))

        if args.free_early:
            # Drop every reference immediately so the allocator is free to
            # reuse the pages while the socket threads may still hold pointers.
            x = None
            back = None if r == 0 else None
            got = None
            mx.clear_cache()

        if i % 25 == 0:
            mark(f"round {i} ok (elems {elems})")

    mark(f"RESULT: {args.rounds} rounds with no error")


if __name__ == "__main__":
    main()
