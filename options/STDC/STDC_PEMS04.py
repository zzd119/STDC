import argparse

class STDC_PEMS04():

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def add_options_specific_arguments(parent_parser):
        temp_args, _ = parent_parser.parse_known_args()
        parser = argparse.ArgumentParser(parents=[parent_parser],add_help=False)
        parser.add_argument("--steps_per_day", type=int,default=288)
        parser.add_argument("--num_nodes", default=307)
        parser.add_argument("--input_len", default=temp_args.history_seq_len)
        parser.add_argument("--output_len", default=temp_args.future_seq_len)
        parser.add_argument("--input_dim", default=3)
        parser.add_argument("--emb_dim", default=64)
        parser.add_argument("--num_layer", default=3)
        parser.add_argument("--droupt", default=0.15)
        parser.add_argument("--train_batch_size", default=32)
        parser.add_argument("--val_batch_size", default=64)
        parser.add_argument("--test_batch_size", default=64)
        parser.add_argument("--forward_features", default=[0, 1, 2])
        parser.add_argument("--target_features", default=[0])
        parser.add_argument("--learning_rate", default=0.001)
        parser.add_argument("--weight_decay", default=0.0001)
        return parser
