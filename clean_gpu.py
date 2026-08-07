import gc
import subprocess
import sys

import torch


def _get_nvidia_smi_memory():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return float(out) / 1024
    except Exception:
        return None


def _get_nvidia_smi_processes():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if not out:
            return []
        processes = []
        for line in out.split("\n"):
            parts = line.split(",")
            if len(parts) >= 2:
                processes.append((parts[0].strip(), float(parts[1].strip()) / 1024))
        return processes
    except Exception:
        return None


def main():
    if not torch.cuda.is_available():
        print("No CUDA device found")
        return

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"PyTorch view:")
    print(f"  allocated: {allocated:.2f} GiB")
    print(f"  reserved:  {reserved:.2f} GiB")
    print(f"  peak:      {peak:.2f} GiB")

    nvsmi_mem = _get_nvidia_smi_memory()
    if nvsmi_mem is not None:
        print(f"\nnvidia-smi reports: {nvsmi_mem:.2f} GiB in use")

    processes = _get_nvidia_smi_processes()
    if processes:
        print(f"\nProcesses using GPU memory:")
        for pid, mem in processes:
            print(f"  PID {pid}: {mem:.2f} GiB")
    elif processes is not None:
        print("\nNo processes listed in nvidia-smi")

    if nvsmi_mem is not None and allocated < 0.1 and nvsmi_mem > 1.0:
        if processes:
            print("\n--- GPU memory held by live process(es) ---")
            print("To free this memory, kill the process(es):")
            for pid, mem in processes:
                print(f"  kill {pid}    ({mem:.2f} GiB)")
            if len(processes) == 1:
                print(f"\nRun:  kill {processes[0][0]}")
                print(f"  Or:  kill -9 {processes[0][0]}  (force kill)")
            else:
                pids = " ".join(p[0] for p in processes)
                print(f"\nRun:  kill {pids}")
                print(f"  Or:  kill -9 {pids}  (force kill)")
        else:
            print("\n--- GPU memory is ORPHANED ---")
            print("The memory was allocated by a previous process that exited or crashed")
            print("without properly releasing CUDA resources.\n")
            print("This memory cannot be freed from a new Python process.\n")
            print("Trying force-allocation trick to trigger driver reclamation...")

            try:
                free, total = torch.cuda.mem_get_info()
                size = int(free * 0.95)
                print(f"  Free memory: {free / 1024**3:.2f} GiB, attempting allocation of {size / 1024**3:.2f} GiB")
                t = torch.empty(size, dtype=torch.uint8, device="cuda")
                del t
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                nvsmi_after = _get_nvidia_smi_memory()
                if nvsmi_after is not None:
                    print(f"  nvidia-smi after force-allocation: {nvsmi_after:.2f} GiB in use")
                    if nvsmi_after < nvsmi_mem * 0.5:
                        print("  Force-allocation succeeded in freeing orphaned memory!")
                    else:
                        print("  Force-allocation did not help.\n")
                        print("Options to free orphaned GPU memory:")
                        print("  1. sudo nvidia-smi -r          (GPU reset, requires root)")
                        print("  2. sudo fuser -k /dev/nvidia*  (kill processes holding GPU)")
                        print("  3. reboot the machine")
                        print("  4. Or prevent orphaned memory by adding proper cleanup")
                        print("     to your training script (see main.py finally block)")
            except RuntimeError as e:
                print(f"  Allocation failed: {e}")
                print("  GPU memory is too fragmented to recover without reset.")

    if allocated < 0.1 and reserved < 0.1 and (nvsmi_mem is None or nvsmi_mem < 0.1):
        print("\nGPU is clean — all memory released")

    sys.exit(0)


if __name__ == "__main__":
    main()
