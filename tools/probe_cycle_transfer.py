"""Diagnostic-only legacy slew comparison, never a formal training release."""
import pathlib,sys
s=pathlib.Path('train_cycle.py');code=s.read_text();assert '--probe' in sys.argv and '--iterations' not in sys.argv
code=code.replace('choices=[120.,240.,480.]','choices=[120.,240.,480.,900.]')
exec(compile(code,str(s),'exec'),{'__name__':'__main__','__file__':str(s)})
