import datetime
import prettytable
from collections import defaultdict
import numpy as np
import torch


@torch.no_grad()
def print_results(multi_eval_aggr_result, title='image-text retrieval results'):
	mean_result, std_result = defaultdict(), defaultdict()
	for k, v in multi_eval_aggr_result.items(): mean_result[k], std_result[k] = round(np.mean(v), 2), round(np.std(v), 2)

	# print(f"[blue]{title}[/blue]")
	results_table = prettytable.PrettyTable()
	results_table.float_format = ".2f"
	results_table.align = 'c'
	results_table.border = True
	results_table.field_names = ['Img R@1', 'Img R@5', 'Img R@10', 'Img R_Mean', 
								 'Txt R@1', 'Txt R@5', 'Txt R@10', 'Txt R_Mean', 
								 'R_Mean']
	results_table.title = title

	if prettytable.__version__ >= '3.16.0':
		results_table.add_divider()
	else:
		results_table.add_row(['-'*10]*9)         

	# mean
	results_table.add_row(
		[f"{mean_result['img_r1']:.2f}", f"{mean_result['img_r5']:.2f}", f"{mean_result['img_r10']:.2f}", f"{mean_result['img_r_mean']:.2f}", 
		 f"{mean_result['txt_r1']:.2f}", f"{mean_result['txt_r5']:.2f}", f"{mean_result['txt_r10']:.2f}", f"{mean_result['txt_r_mean']:.2f}", 
		 f"{mean_result['r_mean']:.2f}"]
		)
	# std
	results_table.add_row(
		# \u00B1
		[f"± {std_result['img_r1']:.2f}", f"± {std_result['img_r5']:.2f}", f"± {std_result['img_r10']:.2f}", f"± {std_result['img_r_mean']:.2f}", 
		f"± {std_result['txt_r1']:.2f}", f"± {std_result['txt_r5']:.2f}", f"± {std_result['txt_r10']:.2f}", f"± {std_result['txt_r_mean']:.2f}", 
		f"± {std_result['r_mean']:.2f}"]
		)
	
	return results_table


def make_timestamp(prefix: str="", suffix: str="") -> str:
	KST_TIMEZONE = 9
	tmstamp = datetime.datetime.now() + datetime.timedelta(hours=KST_TIMEZONE)
	tmstamp = '{:%m_%d_%Y_%H%M%S}'.format(tmstamp)
	
	return prefix + tmstamp + suffix
