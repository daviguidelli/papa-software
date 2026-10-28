"""Live load cell readout. Standalone: needs only python3-lgpio.

  python3 scripts/pi/02_loadcell.py                 # raw counts, auto-tared
  python3 scripts/pi/02_loadcell.py --scale 450.2   # grams, once you know counts per gram

Wiring: HX711 DT -> GPIO5 (pin 29), SCK -> GPIO6 (pin 31), VCC -> 3.3 V, GND -> GND.
Nothing on the cell while it tares. Ctrl+C to stop.
"""
import argparse
import statistics
import sys
import time
from collections import deque


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dat", type=int, default=5)
    ap.add_argument("--clk", type=int, default=6)
    ap.add_argument("--scale", type=float, default=0.0, help="counts per gram")
    ap.add_argument("--tare-samples", type=int, default=20)
    args = ap.parse_args()

    try:
        import lgpio
    except ImportError:
        print("FAIL: sudo apt install -y python3-lgpio")
        return 1

    h = lgpio.gpiochip_open(0)
    lgpio.gpio_claim_input(h, args.dat)
    lgpio.gpio_claim_output(h, args.clk, 0)

    def read() -> int:
        t = time.time() + 1.0
        while lgpio.gpio_read(h, args.dat) != 0:
            if time.time() > t:
                raise TimeoutError("HX711 not responding: check DT/SCK wiring, power and ground")
            time.sleep(0.001)
        v = 0
        for _ in range(24):
            lgpio.gpio_write(h, args.clk, 1)
            lgpio.gpio_write(h, args.clk, 0)
            v = (v << 1) | lgpio.gpio_read(h, args.dat)
        lgpio.gpio_write(h, args.clk, 1)      # 25th pulse: channel A, gain 128
        lgpio.gpio_write(h, args.clk, 0)
        return v - (1 << 24) if v & 0x800000 else v

    try:
        print("taring, keep the cell unloaded...")
        zero = statistics.median(read() for _ in range(args.tare_samples))
        print(f"tare = {zero:.0f} counts. Press gently on the cell. Ctrl+C to stop.\n")
        recent = deque(maxlen=20)
        peak = 0.0
        t0 = time.time()
        n = 0
        while True:
            raw = read()
            n += 1
            d = raw - zero
            val = d / args.scale if args.scale else d
            unit = "g" if args.scale else "counts"
            recent.append(val)
            peak = max(peak, abs(val))
            noise = statistics.pstdev(recent) if len(recent) > 2 else 0.0
            span = peak or 1.0
            bar = "#" * int(30 * min(1.0, max(0.0, val) / span))
            rate = n / (time.time() - t0)
            sys.stdout.write(
                f"\r{val:+10.2f} {unit:6} raw {raw:9d}  noise {noise:6.2f}  {rate:4.1f}/s  |{bar:<30}|"
            )
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("\nstopped")
    except TimeoutError as exc:
        print(f"\nFAIL: {exc}")
        return 1
    finally:
        lgpio.gpiochip_close(h)
    return 0


if __name__ == "__main__":
    sys.exit(main())
