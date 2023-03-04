import os
import json 
import pickle
import time

from utils import get_data, ConfigHandler, pot_detect
from model_uasd import USAD

import numpy as np


from multiprocessing import Process, Lock, Barrier, Queue
from sklearn.cluster import AgglomerativeClustering


def FLAggregate(models):
    global_model = {'share_encoder': None, 'decoder_G': None, 'decoder_D': None}
    total_num = 0
    
    for model in models:
        if global_model['share_encoder'] is None:
            global_model['share_encoder'] = model._shared_encoder.get_weights()
            for i in range(len(global_model['share_encoder'])):
                global_model['share_encoder'][i] = model._train_num * model._shared_encoder.get_weights()[i]
        else:
            for i in range(len(global_model['share_encoder'])):
                global_model['share_encoder'][i] += model._train_num * model._shared_encoder.get_weights()[i]

        if global_model['decoder_G'] is None:
            global_model['decoder_G'] = model._decoder_G.get_weights()
            for i in range(len(global_model['decoder_G'])):
                global_model['decoder_G'][i] = model._train_num * model._decoder_G.get_weights()[i]
        else:
            for i in range(len(global_model['decoder_G'])):
                global_model['decoder_G'][i] += model._train_num * model._decoder_G.get_weights()[i]

        if global_model['decoder_D'] is None:
            global_model['decoder_D'] = model._decoder_D.get_weights()
            for i in range(len(global_model['decoder_D'])):
                global_model['decoder_D'][i] = model._train_num * model._decoder_D.get_weights()[i]
        else:
            for i in range(len(global_model['decoder_D'])):
                global_model['decoder_D'][i] += model._train_num * model._decoder_D.get_weights()[i]
        
        total_num += model._train_num

    global_model['share_encoder'] = [p / total_num for p in global_model['share_encoder']]
    global_model['decoder_G'] = [p / total_num for p in global_model['decoder_G']]
    global_model['decoder_D'] = [p / total_num for p in global_model['decoder_D']]

    return global_model


def run(config, is_main, global_epoch, participants, cluster_message):

    if is_main:
        aggregate_time = 0
        start_time = time.time()

        with_load = 0
        without_load = 0
        if global_epoch == 0:
            for n in range(int(cluster_message['cluster_num'])):
                global_model = USAD(x_dims=config.x_dims, z_dims=config.z_dims, window_size=config.window_size)
                global_model.build()
                global_model.save(os.path.join(config.save_dir, f'global_{n}_{global_epoch}'))
        else:
            for n in range(int(cluster_message['cluster_num'])):
                load_start = time.time()
                global_model = USAD(x_dims=config.x_dims, z_dims=config.z_dims, window_size=config.window_size)
                global_model.restore(os.path.join(config.save_dir, f'global_{n}_{global_epoch-1}'))
                global_model.build()

                local_models = []
                model_participants = []
                for participant in participants:
                    if cluster_message['clusters'][str(participant)] != str(n):
                        continue

                    with open(os.path.join(config.save_dir, f'local_{participant}_{global_epoch-1}/message.json'), 'r') as f:
                        train_num = json.load(f)['train_num']

                    local_model = USAD(x_dims=config.x_dims, z_dims=config.z_dims, window_size=config.window_size, train_num=train_num)
                    local_model.restore(os.path.join(config.save_dir, f'local_{participant}_{global_epoch-1}'))
                    local_model.build()
                    local_models.append(local_model)

                aggregate_start = time.time()
                new_model_parameters = FLAggregate(local_models)
                global_model._shared_encoder.set_weights(new_model_parameters['share_encoder'])
                global_model._decoder_G.set_weights(new_model_parameters['decoder_G'])
                global_model._decoder_D.set_weights(new_model_parameters['decoder_D'])

                global_model.save(os.path.join(config.save_dir, f'global_{n}_{global_epoch}'))
                with_load += time.time() - load_start
                without_load += time.time() - aggregate_start

        aggregate_time += time.time() - start_time
        with open('time_record.txt', 'a') as f:
            f.write(f'epoch: {global_epoch} total aggregate: {aggregate_time} | with load time: {with_load} | without load time: {without_load} \n')
    else:
        train_start = time.time()
        for participant in participants:
            cluster_id = cluster_message['clusters'][str(participant)]
            local_model = USAD(x_dims=config.x_dims, z_dims=config.z_dims, window_size=config.window_size, max_epochs=config.local_max_epoch, message=f'local-{id}')
            local_model.restore(os.path.join(config.save_dir, f'global_{cluster_id}_{global_epoch}'))

            dataset = ''
            for cluster in range(1, 32):
                if os.path.exists(f'./{config.prefix}/{config.dataset}-{cluster}-{participant}_train.pkl'):
                    dataset = f'{config.dataset}-{cluster}-{participant}'

            (x_train, _), (x_test, y_test) = get_data(dataset, do_preprocess=True, prefix=f'./{config.prefix}', x_dims=config.x_dims)

            local_model.fit(x_train)

            result_dir = os.path.join(config.result_dir, f'local_{participant}')
            if not os.path.exists(result_dir):
                os.mkdir(result_dir)

            local_model.save(os.path.join(config.save_dir, f'local_{participant}_{global_epoch}'))

            with open(os.path.join(config.save_dir, f'local_{participant}_{global_epoch}/message.json'), 'w') as f:
                json.dump({'train_num': x_train.shape[0]}, f)

        train_time = time.time() - train_start
        print(f'epoch {global_epoch} : each device time: {train_time / len(participants)}')


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


def main(config):
    
    global_participants = []
    temp = []
    for f in os.listdir(config.prefix):
        if f[-10:] == '_train.pkl':
            global_participants.append(int(f[:-10].split('-')[-1]))
    global_participants.sort()

    participants_group = divide_process_group(global_participants, config.process_num)

    cluster_message = json.load(open(os.path.join(config.cluster_save_dir, f'participants_{config.cluster_distance}_{config.cluster_model_use}_{config.cluster_num}.json'), 'rb'))

    config.save_dir = config.save_dir + f'/{config.cluster_model_use}_{config.cluster_num}'
    config.result_dir = config.result_dir + f'/{config.cluster_model_use}_{config.cluster_num}'
    
    if not os.path.exists(config.save_dir):
        os.mkdir(config.save_dir)
    if not os.path.exists(config.result_dir):
        os.mkdir(config.result_dir)

    global_epoch = 0
    while global_epoch < config.global_max_epoch:

        main_process = Process(target=run, args=(config, True, global_epoch, global_participants, cluster_message))
        main_process.start()
        main_process.join()

        process = [Process(target=run, args=(config, False, global_epoch, participants, cluster_message)) for participants in participants_group]
        for p in process:
            p.start()
        for p in process:
            p.join()

        print(global_epoch + 1, 'finish')
        global_epoch += 1


if __name__ == '__main__':

    config = ConfigHandler().config
    start = time.time()
    main(config)
    print(time.time() - start)
