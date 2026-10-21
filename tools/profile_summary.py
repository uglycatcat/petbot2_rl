"""Summarize completed short-run throughput and whole-card memory peaks."""
import argparse,json,pathlib,re,statistics
p=argparse.ArgumentParser();p.add_argument('logs',nargs='+',type=pathlib.Path);a=p.parse_args()
for log in a.logs:
 text=re.sub(r'\x1b\[[0-9;]*m','',log.read_text(errors='replace'))
 times=[float(x) for x in re.findall(r'Iteration time:\s*([0-9.]+)s',text)]
 exit_file=log.with_suffix('.exit.json');status=json.loads(exit_file.read_text()) if exit_file.exists() else {}
 print(json.dumps({'log':str(log),'iterations':len(times),'median_iteration_s':statistics.median(times[2:]) if len(times)>2 else None,'exit_code':status.get('exit_code'),'peak_card_mib':status.get('sampled_peak_mib')}))
