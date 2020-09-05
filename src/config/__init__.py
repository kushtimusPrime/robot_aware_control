import argparse


def str2bool(v):
    return v.lower() == "true"


def str2intlist(value):
    if not value:
        return value
    else:
        return [int(num) for num in value.split(",")]


def str2list(value):
    if not value:
        return value
    else:
        return [num for num in value.split(",")]


def create_parser():
    """
    Creates the argparser.  Use this to add additional arguments
    to the parser later.
    """
    parser = argparse.ArgumentParser(
        "Robot Aware Cost",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    add_method_arguments(parser)

    return parser


def add_method_arguments(parser):
    # method arguments
    parser.add_argument("--reward_type", type=str, default="weighted", choices=["weighted", "dense", "sparse"])
    parser.add_argument("--robot_pixel_weight", type=str, default=0)

    # control algorithm
    parser.add_argument(
        "--algo",
        type=str,
        default="cem",
        choices=["cem"],
    )

    # training
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num_episodes", type=int, default=100)

    # environment
    parser.add_argument("--env", type=str, default="FetchPush")
    args, unparsed = parser.parse_known_args()
    # arguments specific to algorithms
    if args.algo == "cem":
        add_cem_arguments(parser)

    # env specific args
    if args.env == "FetchPush":
        add_fetch_push_arguments(parser)

    return parser

# Env Hyperparameters
def add_fetch_push_arguments(parser):
    parser.add_argument("--img_dim", type=int, default=128)
    parser.add_argument("--camera_name", type=str, default="external_camera_0", choices=['head_camera_rgb', 'gripper_camera_rgb', 'lidar', 'external_camera_0'])
    parser.add_argument("--pixels_ob", type=str2bool, default=True)
    parser.add_argument("--object_dist_threshold", type=float, default=0.05)
    parser.add_argument("--gripper_dist_threshold", type=float, default=0.025)

# Algo Hyperparameters
def add_cem_arguments(parser):
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--opt_iter", type=int, default=10)
    parser.add_argument("--action_candidates", type=int, default=128)
    parser.add_argument("--topk", type=int, default=5)


def argparser():
    """ Directly parses the arguments. """
    parser = create_parser()
    args, unparsed = parser.parse_known_args()
    assert len(unparsed) == 0
    return args, unparsed