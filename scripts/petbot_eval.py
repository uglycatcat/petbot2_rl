"""Legacy evaluator retired: it silently dropped config fields and biased episode sampling.

Use scripts/petbot_sweep2_eval.py for paired, first-episode evaluation.
Historical reproduction must use the original immutable experiment checkout.
"""
if __name__ == "__main__":
    raise SystemExit("Legacy evaluator disabled. Use scripts/petbot_sweep2_eval.py --checkpoint ... "
                     "--episodes 512 --output ... ; use the original checkout for historical reproduction.")
