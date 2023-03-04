import os
import json 
import pickle
import time


from utils import get_data, ConfigHandler
from model_usad import USAD
from evaluate import get_score

import numpy as np

from sklearn.cluster import AgglomerativeClustering


def caculate_distance(models=None, global_model=None, start=None, end=None, result=None, load_path=None, total_num=None, ctype='full', dtype='Cos'):

    def caculateEucliDistance(m1, m2, global_model=None, ctype='full'):
        dis = 0

        if ctype == 'full' or ctype == 'encoder':
            encoder_1 = m1._encoder.get_weights()
            encoder_2 = m2._encoder.get_weights()
            for n in range(len(encoder_1)):
                if len(encoder_1[n].shape) > 1:
                    dis += np.sum(np.square(encoder_1[n] - encoder_2[n]))

        if ctype == 'full' or ctype == 'decoder':
            decoder_1 = m1._decoder.get_weights()
            decoder_2 = m2._decoder.get_weights()
            for n in range(len(decoder_1)):
                if len(decoder_1[n].shape) > 1:
                    dis += np.sum(np.square(decoder_1[n] - decoder_2[n]))

        return np.sqrt(dis)

    def caculateManhDistance(m1, m2, global_model=None, ctype='full'):
        dis = 0

        if ctype == 'full' or ctype == 'encoder':
            encoder_1 = m1._encoder.get_weights()
            encoder_2 = m2._encoder.get_weights()
            for n in range(len(encoder_1)):
                if len(encoder_1[n].shape) > 1:
                    dis += np.sum(abs(encoder_1[n] - encoder_2[n]))
        
        if ctype == 'full' or ctype == 'decoder':
            decoder_1 = m1._decoder.get_weights()
            decoder_2 = m2._decoder.get_weights()
            for n in range(len(decoder_1)):
                if len(decoder_1[n].shape) > 1:
                    dis += np.sum(abs(decoder_1[n] - decoder_2[n]))

        return dis

    def caculateCosDistance(m1, m2, global_model=None, ctype='full'):
        
        top = 0
        bot_l, bot_r = 0, 0
        if global_model is None:
            if ctype == 'full' or ctype == 'encoder':
                encoder_1 = m1._encoder.get_weights()
                encoder_2 = m2._encoder.get_weights()
                for n in range(len(encoder_1)):
                    if len(encoder_1[n].shape) > 1:
                        top += np.dot(encoder_1[n].reshape(-1), encoder_2[n].reshape(-1))
                        bot_l += np.dot(encoder_1[n].reshape(-1), encoder_1[n].reshape(-1))
                        bot_r += np.dot(encoder_2[n].reshape(-1), encoder_2[n].reshape(-1))
            
            if ctype == 'full' or ctype == 'decoder':
                decoder_1 = m1._decoder.get_weights()
                decoder_2 = m2._decoder.get_weights()
                for n in range(len(decoder_1)):
                    if len(decoder_1[n].shape) > 1:
                        top += np.dot(decoder_1[n].reshape(-1), decoder_2[n].reshape(-1))
                        bot_l += np.dot(decoder_1[n].reshape(-1), decoder_1[n].reshape(-1))
                        bot_r += np.dot(decoder_2[n].reshape(-1), decoder_2[n].reshape(-1))
        else:
            if ctype == 'full' or ctype == 'encoder':
                encoder_1 = m1._encoder.get_weights()
                encoder_2 = m2._encoder.get_weights()
                encoder_g = global_model._encoder.get_weights()
                for n in range(len(encoder_1)):
                    if len(encoder_1[n].shape) > 1:
                        top += np.dot(encoder_1[n].reshape(-1) - encoder_g[n].reshape(-1), encoder_2[n].reshape(-1) - encoder_g[n].reshape(-1))
                        bot_l += np.dot(encoder_1[n].reshape(-1) - encoder_g[n].reshape(-1), encoder_1[n].reshape(-1) - encoder_g[n].reshape(-1))
                        bot_r += np.dot(encoder_2[n].reshape(-1) - encoder_g[n].reshape(-1), encoder_2[n].reshape(-1) - encoder_g[n].reshape(-1))

            if ctype == 'full' or ctype == 'decoder':
                decoder_1 = m1._decoder.get_weights()
                decoder_2 = m2._decoder.get_weights()
                decoder_g = global_model._decoder.get_weights()
                for n in range(len(decoder_1)):
                    if len(decoder_1[n].shape) > 1:
                        top += np.dot(decoder_1[n].reshape(-1) - decoder_g[n].reshape(-1), decoder_2[n].reshape(-1) - decoder_g[n].reshape(-1))
                        bot_l += np.dot(decoder_1[n].reshape(-1) - decoder_g[n].reshape(-1), decoder_1[n].reshape(-1) - decoder_g[n].reshape(-1))
                        bot_r += np.dot(decoder_2[n].reshape(-1) - decoder_g[n].reshape(-1), decoder_2[n].reshape(-1) - decoder_g[n].reshape(-1))
           
        return 1 - top / np.sqrt(bot_l * bot_r)

    if models is None:
        distances = [[0 for j in range(total_num)] for i in range(total_num)]
        global_model = AE(x_dims=config.x_dims, z_dims=config.cluster_z_dims, window_size=config.cluster_window_size)
        global_model.restore(os.path.join(f'{load_path}', f'global'))
        global_model.build()
    else:
        distances = [[0 for j in range(len(models))] for i in range(len(models))]

    if start is None:
        for i in range(len(models)):
            if models is None:
                model_1 = AE(x_dims=config.x_dims, z_dims=config.cluster_z_dims, window_size=config.cluster_window_size)
                model_1.restore(os.path.join(f'{load_path}', f'local_{i}'))
                model_1.build()
            else:
                model_1 = models[i]
            for j in range(i+1, len(models)):
                if models is None:
                    model_2 = AE(x_dims=config.x_dims, z_dims=config.cluster_z_dims, window_size=config.cluster_window_size)
                    model_2.restore(os.path.join(f'{load_path}', f'local_{j}'))
                    model_2.build()
                else:
                    model_2 = models[j]
                
                if dtype == 'Cos':
                    dis = caculateCosDistance(model_1, model_2, global_model, ctype)
                elif dtype == 'Eucli':
                    dis = caculateEucliDistance(model_1, model_2, global_model, ctype)
                elif dtype == 'Manh':
                    dis = caculateManhDistance(model_1, model_2, global_model, ctype)

                distances[i][j] = dis
                distances[j][i] = dis
    else:
        for i in range(start[0], end[0]):
            if models is None:
                model_1 = AE(x_dims=config.x_dims, z_dims=config.cluster_z_dims, window_size=config.cluster_window_size)
                model_1.restore(os.path.join(f'{load_path}', f'local_{i}'))
                model_1.build()
            else:
                model_1 = models[i]
            for j in range(start[1], end[1]):
                if j <= i:
                    continue

                if models is None:
                    model_2 = AE(x_dims=config.x_dims, z_dims=config.cluster_z_dims, window_size=config.cluster_window_size)
                    model_2.restore(os.path.join(f'{load_path}', f'local_{j}'))
                    model_2.build()
                else:
                    model_2 = models[j]
     
                if dtype == 'Cos':
                    dis = caculateCosDistance(model_1, model_2, global_model, ctype)
                elif dtype == 'Eucli':
                    dis = caculateEucliDistance(model_1, model_2, global_model, ctype)
                elif dtype == 'Manh':
                    dis = caculateManhDistance(model_1, model_2, global_model, ctype)

                distances[i][j] = dis
                distances[j][i] = dis
    
    if result is None:
        return distances
    else:
        result.put(np.array(distances))


def divide_process_group(total_participants, group_num):
    num_group = int(len(total_participants) / group_num)

    participants_group = []
    index = 0
    for i in range(group_num):
        if i < len(total_participants) - num_group * group_num:
            participants_group.append(total_participants[index: index+num_group+1])
            index += num_group + 1
        else:
            participants_group.append(total_participants[index: index + num_group])
            index += num_group
    
    return participants_group


def init_cluster(config, total_participants, participants_group):
    cluster_result = Queue()
    main_barrier = Barrier(config.process_num + 1)
    result_barrier = Barrier(config.process_num + 1)
    main_cluster_process = Process(target=cluster, args=(config, True, total_participants, main_barrier, result_barrier, cluster_result))
    main_cluster_process.start()
    
    cluster_process = [Process(target=cluster, args=(config, False, participants, main_barrier, result_barrier, cluster_result)) for participants in participants_group]
    for p in cluster_process:
        p.start()
    for p in cluster_process:
        p.join()

    main_cluster_process.join()

    model, clusters = cluster_result.get()

    participants_cluster_result = {}
    participants_cluster = {}
    for i in range(len(total_participants)):
        participants_cluster[str(total_participants[i])] = str(clusters[i])
    
    participants_cluster_result['clusters'] = participants_cluster
    participants_cluster_result['cluster_num'] = str(model.n_clusters_)
    with open(os.path.join(config.cluster_save_dir, f'participants_{config.cluster_distance}_{config.cluster_model_use}_{config.cluster_num}.json'), 'w') as f:
        json.dump(participants_cluster_result, f)


def cluster(config, participants, result):
    
    if is_main:
        global_model = AE(x_dims=config.x_dims, z_dims=config.cluster_z_dims, window_size=config.cluster_window_size)
        global_model.build()
        global_model.save(os.path.join(config.cluster_save_dir, f'global'))
        main_barrier.wait()

        result_barrier.wait()
        
        distance_queue = Queue()
        caculate_group = []
        total_num = len(participants)

        part_num = 12
        for i in range(part_num):
            for j in range(part_num):
                if i <= j:
                    caculate_group.append([[i*int(total_num/part_num), j*int(total_num/part_num)],[(i+1)*int(total_num/part_num) if i != part_num - 1 else total_num,
                                    (j+1)*int(total_num/part_num) if j != part_num - 1 else total_num]])
        
        local_models = []
        for participant in participants:
            local_model = AE(x_dims=config.x_dims, max_epochs=config.cluster_max_epoch, z_dims=config.cluster_z_dims, window_size=config.cluster_window_size)
            local_model.restore(os.path.join(config.cluster_save_dir, f'local_{participant}'))
            local_model.build()
            local_models.append(local_model)

        distances = caculate_distance(local_models, None, group[0], group[1], distance_queue, None, None, config.cluster_model_use, config.cluster_distance)

        np.save(os.path.join(config.cluster_save_dir, f'distances_{config.cluster_distance}_{config.cluster_model_use}.npy'), distances)

        if config.cluster_num is None:
            if config.cluster_method == 'Hierarchical':
                clusterModel = AgglomerativeClustering(affinity='precomputed', linkage='average', n_clusters=None, distance_threshold=config.cluster_distance_threshold)
                cluster = clusterModel.fit_predict(distances)
        else:
            if config.cluster_method == 'Hierarchical':
                clusterModel = AgglomerativeClustering(affinity='precomputed', linkage='average', n_clusters=config.cluster_num)
                cluster = clusterModel.fit_predict(distances)

        cluster = clusterModel.fit_predict(distances)
        
        result.put((clusterModel, cluster))
    else:
        main_barrier.wait()

        for participant in participants:

            local_model = AE(x_dims=config.x_dims, z_dims=config.cluster_z_dims, max_epochs=config.local_max_epoch, window_size=config.cluster_window_size)
            local_model.restore(os.path.join(config.cluster_save_dir, f'global'))
            local_model.build()

            for cluster in range(1, 32):
                if os.path.exists(f'./{config.prefix}/{config.dataset}-{cluster}-{participant}_train.pkl'):
                    dataset = f'{config.dataset}-{cluster}-{participant}'

            (x_train, _), (x_test, y_test) = get_data(dataset, do_preprocess=False, prefix=f'./{config.prefix}', x_dims=config.x_dims)

            local_model.fit(x_train)
            local_model.save(os.path.join(config.cluster_save_dir, f'local_{participant}'))

        result_barrier.wait()


def main(config):
    
    global_participants = []
    labels = []
    temp = []
    for f in os.listdir(config.prefix):
        if f[-10:] == '_train.pkl':
            global_participants.append(int(f[:-10].split('-')[-1]))
            labels.append(int(f.split('-')[1]))

    pairs = [[global_participants[i], labels[i]] for i in range(len(labels))]
    pairs = sorted(pairs, key=lambda x: x[0])
    labels = [p[1] for p in pairs]

    global_participants = [p[0] for p in pairs]

    participants_group = divide_process_group(global_participants, config.process_num)

    result = init_cluster(config, global_participants, participants_group)


if __name__ == '__main__':

    config = ConfigHandler().config

    main(config)
