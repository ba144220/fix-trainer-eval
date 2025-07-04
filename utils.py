import os
import pandas as pd
from datetime import datetime

def save_results(eval_result, args):
    args_dict = vars(args)
    # Add timestamp YYYY-MM-DD HH:MM:SS
    args_dict["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{args_dict=}")
    if os.path.exists(args.results_csv):
        df = pd.read_csv(args.results_csv)
    else:
        columns = [str(k) for k in args_dict.keys()] + [str(k) for k in eval_result.keys()]
        df = pd.DataFrame(columns=pd.Index(columns))
        
    df = pd.concat([df, pd.DataFrame([{**args_dict, **eval_result}])], ignore_index=True)
    df.to_csv(args.results_csv, index=False)