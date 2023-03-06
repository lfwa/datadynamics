import argparse
import json
import pickle

import numpy as np
import tqdm
import vidmaker

from datacollect.environments import graph_collector_v0
from datacollect.policies import bfs_greedy_policy_v0, premade_policy_v0
from datacollect.utils.post_processing import extract


def main(args):
    with open(args.graph_metadata_file) as f:
        graph_metadata = json.load(f)
        nodes_per_row = graph_metadata["nodes_per_row"]

    with open(args.graph_file, "rb") as f:
        graph = pickle.load(f)

    with open(args.point_labels_file, "rb") as f:
        point_labels = pickle.load(f)

    with open(args.goal_dict_file, "rb") as f:
        goal_dict = pickle.load(f)

    init_agent_labels = [goals[0] for goals in goal_dict.values()]
    max_collect = [len(goals) for goals in goal_dict.values()]

    if args.mode == "screenshot":
        from PIL import Image

        env = graph_collector_v0.env(
            graph=graph,
            point_labels=point_labels,
            init_agent_labels=[0],
            max_collect=[0],
            nodes_per_row=nodes_per_row,
            dynamic_display=True,
            seed=42,
            render_mode="rgb_array",
        )
        env.reset()
        frame = env.render()
        im = Image.fromarray(frame)
        im.save(args.screenshot_output_file)
        return

    if (
        args.mode == "premade"
        or args.cheating_cost_map_file is None
        or args.collection_reward_map_file is None
    ):

        def cheating_cost(point_label):
            return 500 * 0.5

        def collection_reward(point_label):
            return 100

    elif args.mode == "greedy":
        # Load cheating costs and collection rewards that affect greedy policy.
        with open(args.cheating_cost_map_file, "rb") as f:
            cheating_cost_map = pickle.load(f)

        def cheating_cost(point_label):
            return cheating_cost_map[point_label]

        with open(args.collection_reward_map_file, "rb") as f:
            collection_reward_map = pickle.load(f)

        def collection_reward(point_label):
            return collection_reward_map[point_label]

    video = vidmaker.Video(args.video_output_file, late_export=True)

    env = graph_collector_v0.env(
        graph=graph,
        point_labels=point_labels,
        init_agent_labels=init_agent_labels,
        max_collect=max_collect,
        nodes_per_row=nodes_per_row,
        cheating_cost=cheating_cost,
        collection_reward=collection_reward,
        dynamic_display=True,
        seed=42,
        render_mode="rgb_array",
    )

    if args.mode == "premade":
        policy = premade_policy_v0.policy(
            env=env, graph=graph, goal_dict=goal_dict
        )
    elif args.mode == "greedy":
        policy = bfs_greedy_policy_v0.policy(env=env, graph=graph)

    env.reset()
    env.render()

    with tqdm.tqdm(
        total=np.sum(max_collect), desc="Running simulation"
    ) as pbar:
        for agent in env.agent_iter():
            frame = env.render()
            video.update(frame)

            observation, reward, termination, truncation, info = env.last()
            action = policy.action(observation, agent)
            env.step(action)

            if action == -1:
                pbar.update(1)

    video.export(verbose=True)
    extract.save_collections(env=env, filename=args.collections_output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph_file",
        type=str,
        help="File storing environment graph",
        required=True,
    )
    parser.add_argument(
        "--graph_metadata_file",
        type=str,
        help="File storing environment graph metadata",
        required=True,
    )
    parser.add_argument(
        "--point_labels_file",
        type=str,
        help="File storing point labels",
        required=True,
    )
    parser.add_argument(
        "--goal_dict_file",
        type=str,
        help="File storing goal dict",
        required=True,
    )
    parser.add_argument(
        "--mode",
        choices=["greedy", "premade", "screenshot"],
        help="Which policy to use",
        required=True,
    )
    parser.add_argument(
        "--screenshot_output_file",
        type=str,
        help="File to save screenshot to",
        default="screenshot.png",
    )
    parser.add_argument(
        "--video_output_file",
        type=str,
        help="File to save video to",
        default="video.mp4",
    )
    parser.add_argument(
        "--collections_output_file",
        type=str,
        help="File to save collections to",
        default="collections.pkl",
    )
    parser.add_argument(
        "--cheating_cost_map_file",
        type=str,
        help="File storing cheating cost map",
        default=None,
    )
    parser.add_argument(
        "--collection_reward_map_file",
        type=str,
        help="File storing collection reward map",
        default=None,
    )
    args = parser.parse_args()

    main(args)
