import argparse
import numpy as np

def main():
  """ Print a learning rate. """

  description = main.__doc__
  formatter_class = argparse.ArgumentDefaultsHelpFormatter
  parser = argparse.ArgumentParser(description=description, formatter_class=formatter_class)

  parser.add_argument('type', type=str, choices=['grid', 'random'],
                      help='''Learning rate search type. If 'grid', the learning rate will be selected from
                      a grid over [10 ** min, 10 ** max) in log space. If 'random', the learning rate will
                      be selected randomly from [10 ** min, 10 ** max) in log space.''')
  parser.add_argument('log_min', type=float,
                      help='''The minimum possible learning rate in log space.''')
  parser.add_argument('log_max', type=float,
                      help='''The maximum possible learning rate in log space.''')
  parser.add_argument('--grid_log_step', type=float, default=0.25,
                      help='''The grid step size in log space.''')
  parser.add_argument('--grid_ind', type=int, default=0,
                      help='''An index into the learning-rate grid.''')
  args = parser.parse_args()

  if args.type == 'grid':
    learning_rates = 10.0 ** np.arange(args.log_min, args.log_max, args.grid_log_step)
    learning_rate = learning_rates[args.grid_ind]
  elif args.type == 'random':
    learning_rate = 10 ** np.random.uniform(args.log_min, args.log_max)
  else:
    raise ValueError('type must be grid or random.')

  print(learning_rate)


if __name__ == '__main__':
    main()
