import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-b', '--batch_size', default=256, type=int, help='batch_size')
parser.add_argument('-i', '--max_epoch', default=100, type=int, help='max_epoch')
parser.add_argument('-s', '--set_name', default='cifar', type=str, help='set_name')
parser.add_argument('-t', '--task_name', default='hehe', type=str, help='task_name')
parser.add_argument('-r', '--restore', default='', type=str, help='restore')
parser.add_argument('-l', '--code_length', default=32, type=int, help='code_length')