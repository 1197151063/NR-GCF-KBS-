import torch
from parse import parse_args
args = parse_args()


config = {}
config['bpr_batch_size'] = args.bpr_batch
config['K'] = args.K

config['test_u_batch_size'] = args.testbatch

config['epochs'] = args.epochs

config['dataset'] = args.dataset

GPU = torch.cuda.is_available()

device = torch.device('cuda' if GPU else "cpu")

seed = args.seed

dataset = args.dataset

TRAIN_epochs = args.epochs

patience = args.patience

num_neg = args.num_neg

training_objective = args.training_objective

objective_message_dropout = args.objective_message_dropout

adap_tau_mode = args.adap_tau_mode

adap_tau_temperature_2 = args.adap_tau_temperature_2

adap_tau_loss_quantile = args.adap_tau_loss_quantile

adap_tau_recalibration_epoch = args.adap_tau_recalibration_epoch

adap_tau_degree_quantile = args.adap_tau_degree_quantile

adap_tau_initial_positive_gap = args.adap_tau_initial_positive_gap

au_uniformity_weight = args.au_uniformity_weight

au_uniformity_t = args.au_uniformity_t

dropout_rate = args.dropout

decay = args.decay

tau = args.tau

init_weight = args.init_weight

lambda_ = args.lambda_
lr = args.lr
flag = 0
def cprint(words: str):
    print(f"\033[0;30;43m{words}\033[0m")

def bprint(words:str):
    print(f"\033[0;30;45m{words}\033[0m")
