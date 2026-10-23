"""CLI entrypoint for each prototype project."""
import argparse,json
from . import data,core,analytics

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['generate','credit','fraud','drift','segments','features','batch-score','rag','hybrid','sql-init','nl2sql','summary','agent','all'])
    p.add_argument('--question',default='What does fraud policy say about velocity?');p.add_argument('--data',default='data');p.add_argument('--out',default='artifacts');a=p.parse_args()
    commands={'generate':lambda:data.generate(a.data) and {'dataset':'generated'},'credit':lambda:core.credit(a.data,a.out),'fraud':lambda:core.fraud(a.data,a.out),'drift':lambda:core.drift(a.data,a.out),'segments':lambda:core.segment(a.data,a.out),'features':lambda:core.features(a.data,a.out),'batch-score':lambda:core.score_batch(a.data,a.out),'rag':lambda:analytics.rag(a.question,a.data+'/policies'),'hybrid':lambda:analytics.hybrid(a.question,a.data+'/policies'),'sql-init':lambda:{'db':analytics.initialize_sqlite(a.data,a.out+'/analytics.db')},'nl2sql':lambda:analytics.nl2sql(a.question,a.out+'/analytics.db'),'summary':lambda:analytics.summarize(a.out),'agent':lambda:analytics.agent(a.question,a.data,a.out)}
    if a.command=='all':
        order=['generate','credit','fraud','drift','segments','features','batch-score','sql-init','summary']
        print(json.dumps({k:commands[k]() for k in order},indent=2,default=str))
    else:print(json.dumps(commands[a.command](),indent=2,default=str))
if __name__=='__main__':main()
