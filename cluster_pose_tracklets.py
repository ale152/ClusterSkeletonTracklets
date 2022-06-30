import os
import json
import numpy as np
from pathlib import Path
from zipfile import ZipFile
from matplotlib import pyplot as plt


def main(zip_path, output_path, min_occurences, eps_skel, plot_info=True, quiet=False):
    with ZipFile(zip_path) as myzip:
        # Read all files from the zip archive
        all_files = myzip.namelist()
    
        # Filter .json files only
        json_files = [bf for bf in all_files if bf.endswith('.json')]
        # Make sure json files are sorted
        json_files.sort()
    
        # Open json files to read body poses
        json_poses = []
        if not quiet:
            print('Loading json files...')
        for json_file in json_files:
            with myzip.open(json_file) as src:
                data = json.load(src)
                json_poses.append(data['people'])
    
    # Extract joint coordinates (x, y) from the poses. Make a "features" vector with it by appending the frame number
    # at the end of it.
    features = []
    for frame, json_pose in enumerate(json_poses):
        for bf in json_pose:
            joints = bf['pose_keypoints_2d']
            # Remove the joint confidence
            del joints[2::3]
            # Append the frame number at the end of the poses
            joints.append(frame)
            features.append(joints)
    features = np.array(features)
    
    if len(features) == 0:
        if not quiet:
            print(f'No poses found')
        return None

    labels, dbg_distances = cluster_poses(features, min_occurences, eps_skel)

    n_tracklets = labels.max()
    if not quiet:
        print(f'{n_tracklets} tracklets found. Exporting them in independent files')
    output_path.mkdir(exist_ok=True, parents=True)

    if plot_info:
        plt.figure('clusters')
        plt.clf()
    
    for tracklet in range(1, n_tracklets+1):
        n_occ = np.sum(labels == tracklet)
        if n_occ > min_occurences:
            if not quiet:
                print(f'Saving Id #{tracklet}: {n_occ} occurrencies')
            data = features[labels == tracklet, :]
            np.savez_compressed(output_path / f'ID_{tracklet:03d}.npz', data=data)

            if plot_info:
                plt.figure('clusters')
                plt.plot(data[:, 0], data[:, 1], '.')

                plt.figure(f'ID {tracklet}', figsize=[20, 10])
                plt.scatter(data[:, 0:-1:2], data[:, 1:-1:2], 1, np.repeat(data[:, -1, None], 25, axis=1), cmap='jet')
                plt.axis('equal')
                plt.xlim([0, 640])
                plt.ylim([0, 480])
                plt.gca().invert_yaxis()
                plt.savefig(output_path / f'tracklet_{tracklet}.png')

    if plot_info:
        plt.figure('clusters')
        plt.title('Midpoint of skeletons')
        plt.gca().invert_yaxis()
        plt.savefig(output_path / 'clusters.png')

        plt.figure('dbg_distances', figsize=[20, 10])
        plt.clf()
        plt.subplot(1,2,1)
        plt.hist(dbg_distances, 200)
        plt.title('Histogram of distances')
        plt.xlabel('Distance')
        plt.ylabel('Number of occurrencies')
        plt.subplot(1, 2, 2)
        plt.hist(dbg_distances[dbg_distances < 1000], 200)
        plt.title('Capped-Log Histogram of distances')
        plt.xlabel('Distance')
        plt.ylabel('Number of occurrencies')
        plt.yscale('log')
        plt.savefig(output_path / 'dbg_distances.png')


def cluster_poses(features, min_occurences, eps_skel):
    '''Cluster poses together from the features vector'''
    # Initialise labels and classes
    n_features = features.shape[0]
    unclass_skeletons = np.arange(0, n_features)
    labels = np.zeros((n_features), dtype=int)
    class_id = 1

    dbg_distances = []  # Only used for debugging/visualizations

    # Iteratively cluster poses together. Starts with the first unclassified skeleton
    while len(unclass_skeletons) > min_occurences:
        # Assign the current class id to the first unclassified skeleton
        labels[unclass_skeletons[0]] = class_id
        # The last pose is the one we just classified
        last_pose = features[unclass_skeletons[0], :]
        last_mask = last_pose > 0
        last_frnum = features[unclass_skeletons[0], -1]

        # Process the remaining skeletons
        for fr in unclass_skeletons[1:]:
            this_pose = features[fr, :]
            this_mask = this_pose > 0
            this_frnum = features[fr, -1]

            # Calculate the distance between the last pose and the current one only using valid joints
            mask = last_mask * this_mask
            n_mask = np.sum(mask)
            # Make sure the skeletons have at least one joint in common
            if n_mask > 1:
                distance = np.sqrt(np.sum(np.square(last_pose[mask] - this_pose[mask])) / n_mask)
                dbg_distances.append(distance)

                # If the last pose is very close to this one, it probably belongs to the same tracklet
                # Also make sure that the frame number of this pose is different from the one of last pose. If they
                # are the same frnum, they must be different tracklets
                if distance < eps_skel and this_frnum != last_frnum:
                    labels[fr] = class_id
                    last_pose = this_pose.copy()
                    last_mask = this_mask.copy()
                    last_frnum = this_frnum.copy()

        # Remove unclassified frames
        unclass_skeletons = np.where(labels == 0)[0]
        class_id += 1

    dbg_distances = np.array(dbg_distances)
    return labels, dbg_distances


if __name__ == '__main__':
    # Path to the zip file containing the json poses
    zip_path = Path(r"./video_keypoints.zip")
    # Tracklets will be saved in .npz archives in the output folder
    output_path = Path(r"./tracklets")

    # Minimum number of poses for a tracklet to be considered valid
    min_occurences = 60
    # Min distance allowed between two skeletons to be considered the same tracklet (dbg_distances can be used to help
    # find the best threshold
    eps = 50
    main(zip_path, output_path, min_occurences, eps)