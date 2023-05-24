import json
import pickle


def main():
    with open('full_nuscenes_infos_10sweeps_val.pkl', 'rb') as f:
        infos = pickle.load(f)

    print('len(infos): ', len(infos))

    sample_tokens = []
    for info in infos:
        sample_tokens.append(info['token'])
    
    print('num unq sample tokens: ', len(list(set(sample_tokens))))
                
    return 
    with open('results_nusc.json', 'r') as f:
        results = json.load(f)
    
    print(results['meta'])

    print('num sample tokens: ', len(results['results'].keys()))


if __name__ == '__main__':
    main()
