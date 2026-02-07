import os
import pickle
import datetime
import argparse
import numpy as np
import pandas as pd
import h5py

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    auc,
    confusion_matrix,
    precision_recall_curve
)

import pickle as pkl

np.random.seed(0)

class_count_mapping = {
    'cptac_lung': 1,
    'camelyon16': 3,
    'panda': 6,
    'plco_lung': 1,
    'ovarian': 5,
    'tcga_brca': 1,
    'tcga_prad': 4,
    'plco_breast': 3,
    'brca_gene': 1,
    'crc_gene': 1
}

def calculate_metrics(preds, y_true, task_type='binary'):
    preds = np.array(preds)
    y_true = np.array(y_true)
    best_threshold = 0.5

    if task_type == 'binary':
        if len(preds.shape) > 1 and preds.shape[1] == 2:
            preds = preds[:, 1]

        results_arr = (preds > best_threshold).astype(int)
        auc_ = roc_auc_score(y_true, preds)
        precision = precision_score(y_true, results_arr, zero_division=0)
        recall = recall_score(y_true, results_arr, zero_division=0)
        f1 = f1_score(y_true, results_arr, zero_division=0)

        cm = confusion_matrix(y_true, results_arr)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            sensitivity, specificity = 0, 0

        precision_curve, recall_curve, _ = precision_recall_curve(y_true, preds)
        auprc = auc(recall_curve, precision_curve)

        return {
            'auc': auc_,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auprc': auprc,
            'sensitivity': sensitivity,
            'specificity': specificity
        }

    elif task_type == 'multi':
        results = np.argmax(preds, axis=1)
        auc_ = roc_auc_score(y_true, preds, multi_class='ovr', average='macro')
        precision = precision_score(y_true, results, average='macro', zero_division=0)
        recall = recall_score(y_true, results, average='macro', zero_division=0)
        f1 = f1_score(y_true, results, average='macro', zero_division=0)

        return {
            'auc': auc_,
            'precision_macro': precision,
            'recall_macro': recall,
            'f1': f1
        }
    else:
        raise ValueError(f'Invalid task_type: {task_type}')

def ensemble_predictions(all_fold_preds):
    stacked = np.stack(all_fold_preds, axis=0)
    return np.mean(stacked, axis=0)

def save_results_to_csv(results_list, save_path):
    df = pd.DataFrame(results_list)
    df.to_csv(save_path, index=False)
    print(f'Results saved to: {save_path}')
    return df

def load_h5_features(h5_path):
    with h5py.File(h5_path, 'r') as f:
        features = np.array(f['features'])
    return np.mean(features, axis=0)

def load_features(args, names, labels):
    features_list = []
    labels_list = []

    for name, label in zip(names, labels):
        try:
            if args.h5_file:
                feature_file = os.path.join(args.feat_path, f'{name}.h5')
                features = load_h5_features(feature_file)
            else:
                feature_bag = os.path.join(args.feat_path, f'{name}.npy')
                features = np.array(np.load(feature_bag, allow_pickle=True))
                # if features are 2D (tile-level) then mean pool
                if len(features.shape) > 1:
                    features = np.mean(features, axis=0)
        except Exception as e:
            print(f'Warning: missing {name} - {e}')
            continue

        features_list.append(features)
        labels_list.append(label)

    return np.array(features_list), np.array(labels_list)


def parse_fold_data(fold_data):
    if 'train_ids' in fold_data:
        return (
            fold_data['train_ids'],
            fold_data['train_labels'],
            fold_data['val_ids'],
            fold_data['val_labels']
        )
    elif 'train' in fold_data:
        return (
            fold_data['train']['train_ids'],
            fold_data['train']['train_labels'],
            fold_data['val']['val_ids'],
            fold_data['val']['val_labels']
        )
    else:
        raise ValueError(f'Unknown fold data format: {fold_data.keys()}')

def main(args):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    with open(args.cv_split_file, 'rb') as f:
        cv_folds = pkl.load(f)

    if 'fold_0' in cv_folds:
        fold_keys = [f'fold_{i}' for i in range(args.fold_count)]
    else:
        fold_keys = list(range(args.fold_count))

    has_holdout = 'holdout_test' in cv_folds
    print(f'Has holdout test set: {has_holdout}')

    os.makedirs(args.save_root, exist_ok=True)

    all_results = []
    cross_fold_metrics = {}
    fold_val_aucs = []
    fold_val_f1s = []
    fold_models = []

    for fold_idx, fold_key in enumerate(fold_keys):
        print(f'\n--- Fold {fold_idx} ---')

        save_folder = os.path.join(args.save_root, f'checkpoints_fold_{fold_idx}')
        os.makedirs(save_folder, exist_ok=True)

        fold_data = cv_folds[fold_key]
        train_names, train_labels, val_names, val_labels = parse_fold_data(fold_data)

        train_features, train_labels = load_features(args, train_names, train_labels)
        val_features, val_labels = load_features(args, val_names, val_labels)

        print(f'Train: {len(train_features)} slides, Val: {len(val_features)} slides')
        print(f'Feature dim: {train_features.shape[1]}')

        NUM_C = 2
        COST = (train_features.shape[1] * NUM_C) / 100
        clf = LogisticRegression(C=COST, max_iter=10000, verbose=0, random_state=0)
        clf.fit(X=train_features, y=train_labels)

        val_pred_scores = clf.predict_proba(X=val_features)
        val_metrics = calculate_metrics(val_pred_scores, val_labels, task_type=args.task_type)

        print(f'Val AUC: {val_metrics["auc"]:.4f}, Val F1: {val_metrics["f1"]:.4f}')

        fold_val_aucs.append(val_metrics['auc'])
        fold_val_f1s.append(val_metrics['f1'])
        fold_models.append(clf)

        cross_fold_metrics[fold_idx] = {
            'val_auc': val_metrics['auc'],
            'val_f1': val_metrics['f1'],
            'val_metrics': val_metrics
        }

        with open(os.path.join(save_folder, f'fold_{fold_idx}_metrics.pkl'), 'wb') as f:
            pkl.dump(val_metrics, f)

        with open(os.path.join(save_folder, f'model_fold_{fold_idx}.pkl'), 'wb') as f:
            pkl.dump(clf, f)

        all_results.append({
            'task': args.task,
            'model_type': 'linear_probe',
            'feature_type': args.feature_type,
            'phase': 'cv_validation',
            'fold': fold_idx,
            'val_auc': val_metrics['auc'],
            'val_f1': val_metrics['f1'],
            'timestamp': timestamp
        })

    print('CROSS-VALIDATION SUMMARY')
    print(f'CV Validation AUCs: {fold_val_aucs}')
    print(f'Mean CV AUC: {np.mean(fold_val_aucs):.4f} +/- {np.std(fold_val_aucs):.4f}')
    print(f'CV Validation F1s: {fold_val_f1s}')
    print(f'Mean CV F1: {np.mean(fold_val_f1s):.4f} +/- {np.std(fold_val_f1s):.4f}')

    if has_holdout:

        holdout_data = cv_folds['holdout_test']
        test_names = holdout_data['test_ids']
        test_labels_raw = holdout_data['test_labels']

        test_features, test_labels = load_features(args, test_names, test_labels_raw)
        print(f'Holdout test set: {len(test_features)} slides')

        all_fold_preds = []
        individual_aucs = []
        individual_f1s = []

        for fold_idx, clf in enumerate(fold_models):
            fold_preds = clf.predict_proba(X=test_features)
            all_fold_preds.append(fold_preds)

            fold_metrics = calculate_metrics(fold_preds, test_labels, task_type=args.task_type)
            individual_aucs.append(fold_metrics['auc'])
            individual_f1s.append(fold_metrics['f1'])

            print(f'Fold {fold_idx} model - Test AUC: {fold_metrics["auc"]:.4f}, Test F1: {fold_metrics["f1"]:.4f}')

            all_results.append({
                'task': args.task,
                'model_type': 'linear_probe',
                'feature_type': args.feature_type,
                'phase': 'holdout_individual',
                'fold': fold_idx,
                'test_auc': fold_metrics['auc'],
                'test_f1': fold_metrics['f1'],
                'timestamp': timestamp
            })

        ensemble_preds = ensemble_predictions(all_fold_preds)
        ensemble_metrics = calculate_metrics(ensemble_preds, test_labels, task_type=args.task_type)

        print(f'Ensemble Test AUC: {ensemble_metrics["auc"]:.4f}')
        print(f'Ensemble Test F1: {ensemble_metrics["f1"]:.4f}')

        holdout_results = {
            'ensemble_auc': ensemble_metrics['auc'],
            'ensemble_f1': ensemble_metrics['f1'],
            'ensemble_metrics': ensemble_metrics,
            'individual_aucs': individual_aucs,
            'individual_f1s': individual_f1s
        }

        all_results.append({
            'task': args.task,
            'model_type': 'linear_probe',
            'feature_type': args.feature_type,
            'phase': 'holdout_ensemble',
            'fold': 'ensemble',
            'test_holdout_auc': ensemble_metrics['auc'],
            'test_holdout_f1': ensemble_metrics['f1'],
            'timestamp': timestamp
        })
    else:
        holdout_results = None
        ensemble_metrics = None

    results_dict = {
        'cross_fold_metrics': cross_fold_metrics,
        'cv_mean_auc': np.mean(fold_val_aucs),
        'cv_std_auc': np.std(fold_val_aucs),
        'cv_mean_f1': np.mean(fold_val_f1s),
        'cv_std_f1': np.std(fold_val_f1s),
        'holdout_results': holdout_results,
        'args': vars(args)
    }

    with open(os.path.join(args.save_root, f'cross_val_results_{args.task}.pkl'), 'wb') as f:
        pkl.dump(results_dict, f)

    csv_path = os.path.join(args.save_root, f'results_{args.task}.csv')
    save_results_to_csv(all_results, csv_path)

    print(f'Task: {args.task}')
    print(f'Model: Linear Probe (Logistic Regression)')
    print(f'Feature type: {args.feature_type}')
    print(f'CV Mean AUC: {np.mean(fold_val_aucs):.4f} +/- {np.std(fold_val_aucs):.4f}')
    print(f'CV Mean F1: {np.mean(fold_val_f1s):.4f} +/- {np.std(fold_val_f1s):.4f}')
    if holdout_results:
        print(f'Holdout Ensemble AUC: {holdout_results["ensemble_auc"]:.4f}')
        print(f'Holdout Ensemble F1: {holdout_results["ensemble_f1"]:.4f}')
    print(f'\nResults saved to: {args.save_root}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_type', type=str, default='uni')
    parser.add_argument('--cv_split_file', type=str)
    parser.add_argument('--fold_count', type=int)
    parser.add_argument('--h5_file', type = bool, default = False)
    parser.add_argument('--feat_path', type=str)
    parser.add_argument('--save_root', type=str)
    parser.add_argument('--exp_name', type=str, default='linear_probe')
    parser.add_argument('--task', type=str)
    parser.add_argument('--task_type', type=str, default='multi',
                        choices=['binary', 'multi'])

    args = parser.parse_args()
    main(args)
