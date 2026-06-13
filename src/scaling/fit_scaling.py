import pandas as pd
import numpy as np
import re
from scipy.optimize import minimize
from itertools import product


def parse_token_multiplier(token_multiplier_str):
    match = re.match(r'(\d+\.?\d*)', token_multiplier_str)
    if match:
        return float(match.group(1))
    else:
        raise ValueError(f"cannot parse token_multiplier: {token_multiplier_str}")


def load_and_process_data(file_path):

    df = pd.read_csv(file_path, sep='\t')
    
    train_tokens_values = []
    
    for _, row in df.iterrows():
        token_mult = parse_token_multiplier(row['token_multiplier'])
        n_params = row['n']

        train_tokens = token_mult * n_params
        train_tokens_values.append(train_tokens)

    
    df['train_tokens'] = train_tokens_values
    return df


def huber_loss(residuals, epsilon=1e-3):
    abs_residuals = np.abs(residuals)
    quadratic = np.minimum(abs_residuals, epsilon)
    linear = abs_residuals - quadratic
    return np.sum(0.5 * quadratic**2 + epsilon * linear)


def compute_residual_loss(residuals, epsilon=1e-3, loss_fn='huber'):
    if loss_fn == 'mse':
        return np.sum(residuals ** 2)
    else:
        return huber_loss(residuals, epsilon)


def log_sum_exp(x1, x2, x3):
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    x3 = np.asarray(x3)
    max_val = np.maximum(np.maximum(x1, x2), x3)
    return max_val + np.log(np.exp(x1 - max_val) + np.exp(x2 - max_val) + np.exp(x3 - max_val))


def log_sum_exp_2(x1, x2):
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    max_val = np.maximum(x1, x2)
    return max_val + np.log(np.exp(x1 - max_val) + np.exp(x2 - max_val))


def predict_loss(N, D, a, b, c, alpha, beta):
    N = np.asarray(N)
    D = np.asarray(D)
    return a + b / (np.maximum(N, 1e-10) ** alpha) + c / (np.maximum(D, 1e-10) ** beta)


def predict_loss_no_constant(N, D, b, c, alpha, beta):
    N = np.asarray(N)
    D = np.asarray(D)
    return b / (np.maximum(N, 1e-10) ** alpha) + c / (np.maximum(D, 1e-10) ** beta)


def loss_function(params, N, D, y_true_log, epsilon=1e-3, loss_fn='huber'):

    A, B, C, alpha, beta = params
    
    log_N = np.log(np.maximum(N, 1e-10))
    log_D = np.log(np.maximum(D, 1e-10))
    
    term1 = B - alpha * log_N
    term2 = C - beta * log_D
    term3 = np.full_like(term1, A)
    
    y_pred_log = log_sum_exp(term1, term2, term3)
    
    residuals = y_pred_log - y_true_log
    
    return compute_residual_loss(residuals, epsilon, loss_fn)


def loss_function_no_constant(params, N, D, y_true_log, epsilon=1e-3, loss_fn='huber'):
    B, C, alpha, beta = params
    
    log_N = np.log(np.maximum(N, 1e-10))
    log_D = np.log(np.maximum(D, 1e-10))
    
    term1 = B - alpha * log_N
    term2 = C - beta * log_D
    
    y_pred_log = log_sum_exp_2(term1, term2)
    
    residuals = y_pred_log - y_true_log
    
    return compute_residual_loss(residuals, epsilon, loss_fn)


def fit_loss_function(df_fit, epsilon=1e-3, param_field='n_with_emb', loss_fn='huber'):
    N = df_fit[param_field].values
    D = df_fit['train_tokens'].values
    y_true = df_fit['loss'].values
    
    y_true_log = np.log(np.maximum(y_true, 1e-10))
    
    alpha_grid = [0, 0.5, 1, 1.5, 2]
    beta_grid = [0, 0.5, 1.0, 1.5, 2]
    A_grid = [-1, -0.5, 0, 0.5, 1]
    B_grid = [0, 3, 6, 9, 12]
    C_grid = [0, 3, 6, 9, 12]
    
    bounds = [(None, None), (None, None), (None, None), (0, None), (0, None)]
    
    print("Performing grid search for initial parameters...")
    print("(This will optimize each combination to find the best initial parameters)")
    best_final_loss = np.inf
    best_initial = None
    best_optimized_params = None
    total_combinations = len(alpha_grid) * len(beta_grid) * len(A_grid) * len(B_grid) * len(C_grid)
    print(f"Total grid combinations: {total_combinations}")
    
    for idx, (alpha_init, beta_init, A_init, B_init, C_init) in enumerate(
        product(alpha_grid, beta_grid, A_grid, B_grid, C_grid)
    ):
        initial_params = [A_init, B_init, C_init, alpha_init, beta_init]
        
        try:
            result = minimize(
                loss_function,
                initial_params,
                args=(N, D, y_true_log, epsilon, loss_fn),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-6}
            )
            
            final_loss = result.fun
            
            if final_loss < best_final_loss:
                best_final_loss = final_loss
                best_initial = initial_params
                best_optimized_params = result.x
        except:
            continue
        
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{total_combinations} combinations, best final loss so far: {best_final_loss:.6f}")
    
    if best_initial is None:
        raise ValueError("Grid search failed to find valid initial parameters")
    
    print(f"Grid search completed. Best initial parameters:")
    print(f"  A={best_initial[0]:.1f}, B={best_initial[1]:.1f}, C={best_initial[2]:.1f}, "
          f"alpha={best_initial[3]:.1f}, beta={best_initial[4]:.1f}")
    print(f"  Best final loss after optimization: {best_final_loss:.6f}")
    
    print("Starting final optimization with best optimized parameters...")
    result = minimize(
        loss_function,
        best_optimized_params,
        args=(N, D, y_true_log, epsilon, loss_fn),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 10000, 'ftol': 1e-9}
    )
    
    if not result.success:
        print(f"Warning: Optimization did not converge: {result.message}")
    
    A, B, C, alpha, beta = result.x
    
    a = np.exp(A)
    b = np.exp(B)
    c = np.exp(C)
    
    y_pred = predict_loss(N, D, a, b, c, alpha, beta)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return a, b, c, alpha, beta, r_squared


def fit_loss_function_no_constant(df_fit, epsilon=1e-3, param_field='n_with_emb', loss_fn='huber'):
    N = df_fit[param_field].values
    D = df_fit['train_tokens'].values
    y_true = df_fit['loss'].values
    
    y_true_log = np.log(np.maximum(y_true, 1e-10))
    
    alpha_grid = [0.1, 0.2, 0.5, 1, 1.5, 2]
    beta_grid = [0.1, 0.2, 0.5, 0.8, 1.0, 1.2]
    B_grid = [0, 1, 2, 4, 6, 8, 10, 12]
    C_grid = [0, 1, 2, 8, 10, 12]
    
    # adjust the bounds in appropriate range for the best fit
    bounds = [(None, None), (None, None), (0.2, 0.3), (0, None)]

    print("Performing grid search for initial parameters (no constant term)...")
    print("(This will optimize each combination to find the best initial parameters)")
    best_final_loss = np.inf
    best_initial = None
    best_optimized_params = None
    total_combinations = len(alpha_grid) * len(beta_grid) * len(B_grid) * len(C_grid)
    print(f"Total grid combinations: {total_combinations}")
    
    for idx, (alpha_init, beta_init, B_init, C_init) in enumerate(
        product(alpha_grid, beta_grid, B_grid, C_grid)
    ):
        initial_params = [B_init, C_init, alpha_init, beta_init]
        
        try:
            result = minimize(
                loss_function_no_constant,
                initial_params,
                args=(N, D, y_true_log, epsilon, loss_fn),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter':3000, 'ftol': 1e-6}
            )
            
            final_loss = result.fun
            
            if final_loss < best_final_loss:
                best_final_loss = final_loss
                best_initial = initial_params
                best_optimized_params = result.x
        except:
            continue
        
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{total_combinations} combinations, best final loss so far: {best_final_loss:.6f}")
    
    if best_initial is None:
        raise ValueError("Grid search failed to find valid initial parameters")
    
    print(f"Grid search completed. Best initial parameters:")
    print(f"  B={best_initial[0]:.1f}, C={best_initial[1]:.1f}, "
          f"alpha={best_initial[2]:.1f}, beta={best_initial[3]:.1f}")
    print(f"  Best final loss after optimization: {best_final_loss:.6f}")
    
    print("Starting final optimization with best optimized parameters...")
    result = minimize(
        loss_function_no_constant,
        best_optimized_params,
        args=(N, D, y_true_log, epsilon, loss_fn),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 20000, 'ftol': 1e-8}
    )
    
    if not result.success:
        print(f"Warning: Optimization did not converge: {result.message}")
    
    B, C, alpha, beta = result.x
    
    b = np.exp(B)
    c = np.exp(C)
    
    y_pred = predict_loss_no_constant(N, D, b, c, alpha, beta)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return b, c, alpha, beta, r_squared

def compute_fit_quality(df_train, df_val=None, epsilon=1e-3, param_field='n_with_emb', formula_type='default', loss_fn='huber', log_loss_for_ppl=False):
    if len(df_train) < 5:
        print(f"Warning: Need at least 5 points for fitting, got {len(df_train)}")
        return
    
    print(f"Fitting on training set ({len(df_train)} points), formula: {formula_type}...")
    fit_params = {}
    
    if formula_type == 'default':
        a, b, c, alpha, beta, r_squared_train = fit_loss_function(df_train, epsilon=epsilon, param_field=param_field, loss_fn=loss_fn)
        _predict = lambda N, D: predict_loss(N, D, a, b, c, alpha, beta)
        fit_params = {'a': a, 'b': b, 'c': c, 'α': alpha, 'β': beta}
    elif formula_type == 'no_constant':
        b, c, alpha, beta, r_squared_train = fit_loss_function_no_constant(df_train, epsilon=epsilon, param_field=param_field, loss_fn=loss_fn)
        _predict = lambda N, D: predict_loss_no_constant(N, D, b, c, alpha, beta)
        fit_params = {'b': b, 'c': c, 'α': alpha, 'β': beta}
    else:
        raise ValueError(f"Unknown formula_type: {formula_type}. Choose from: default, no_constant")
    
    N_train = df_train[param_field].values
    D_train = df_train['train_tokens'].values
    loss_actual_train = df_train['loss'].values
    loss_predicted_train = _predict(N_train, D_train)
    residuals_train = loss_actual_train - loss_predicted_train
    
    if df_val is not None and len(df_val) > 0:
        N_val = df_val[param_field].values
        D_val = df_val['train_tokens'].values
        loss_actual_val = df_val['loss'].values
        loss_predicted_val = _predict(N_val, D_val)
        residuals_val = loss_actual_val - loss_predicted_val
        
        ss_res_val = np.sum((loss_actual_val - loss_predicted_val) ** 2)
        ss_tot_val = np.sum((loss_actual_val - np.mean(loss_actual_val)) ** 2)
        r_squared_val = 1 - (ss_res_val / ss_tot_val) if ss_tot_val > 0 else 0
        
        print(f"\nValidation set metrics ({len(df_val)} points):")
        print(f"  R² = {r_squared_val:.6f}")
        print(f"  RMSE = {np.sqrt(np.mean(residuals_val**2)):.6f}")
        print(f"  MAE = {np.mean(np.abs(residuals_val)):.6f}")

        val_detail = df_val[['model_size', 'token_multiplier', param_field, 'train_tokens']].copy()
        val_detail['actual'] = loss_actual_val
        val_detail['predicted'] = loss_predicted_val
        val_detail['diff'] = val_detail['actual'] - val_detail['predicted']

        print("\nValidation set details (actual vs predicted):")
        print(val_detail.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

        if log_loss_for_ppl:
            val_detail_ppl = val_detail.copy()
            val_detail_ppl['actual_ppl'] = np.exp(val_detail_ppl['actual'])
            val_detail_ppl['predicted_ppl'] = np.exp(val_detail_ppl['predicted'])
            val_detail_ppl['diff_ppl'] = val_detail_ppl['actual_ppl'] - val_detail_ppl['predicted_ppl']

            print("\nValidation set details in PPL space:")
            print(
                val_detail_ppl[
                    ['model_size', 'token_multiplier', param_field, 'train_tokens', 'actual_ppl', 'predicted_ppl', 'diff_ppl']
                ].to_string(index=False, float_format=lambda x: f"{x:.6f}")
            )

    rmse_train = np.sqrt(np.mean(residuals_train**2))
    mae_train = np.mean(np.abs(residuals_train))
    
    if formula_type == 'default':
        eq_str = f'loss = {fit_params["a"]:.6f} + {fit_params["b"]:.6f}·{param_field}^(-{fit_params["α"]:.6f}) + {fit_params["c"]:.6f}·D^(-{fit_params["β"]:.6f})'
    elif formula_type == 'no_constant':
        eq_str = f'loss = {fit_params["b"]:.6f}·{param_field}^(-{fit_params["α"]:.6f}) + {fit_params["c"]:.6f}·D^(-{fit_params["β"]:.6f})'
    
    print(f"\nFit results (L-BFGS-B with {loss_fn} loss, ε={epsilon}, formula={formula_type}):")
    print(f"  {eq_str}")
    print(f"\nTraining set metrics ({len(df_train)} points):")
    print(f"  R² = {r_squared_train:.6f}")
    print(f"  RMSE = {rmse_train:.6f}")
    print(f"  MAE = {mae_train:.6f}")
    print(f"\nParameters:")
    for pname, pval in fit_params.items():
        print(f"  {pname} = {pval:.6f}")
    
    return fit_params


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fit loss scaling and print fit quality metrics')
    parser.add_argument('--input', type=str, 
                       default='data/scaling/loss_token.dense.tsv')
    parser.add_argument('--epsilon', type=float, 
                       default=1e-3),
    parser.add_argument('--param-field', type=str,
                       default='n'),
    parser.add_argument('--validate', action='store_true'),
    parser.add_argument('--formula', type=str, default='no_constant', choices=['default', 'no_constant']),
    parser.add_argument('--longppl', action='store_true'),
    parser.add_argument('--loss-fn', type=str, default='huber', choices=['huber', 'mse']),
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.input}...")
    df = load_and_process_data(args.input)
    
    formula_type = args.formula

    if args.longppl:
        print(f"\nApplying log transform to loss column for longppl.")
        print(f"  Original loss range: [{df['loss'].min():.4f}, {df['loss'].max():.4f}]")
        df['loss'] = np.log(df['loss'])
        print(f"  After log transform: [{df['loss'].min():.4f}, {df['loss'].max():.4f}]")
    else:
        print(f"  Original loss range: [{df['loss'].min():.4f}, {df['loss'].max():.4f}]")
    
    if args.param_field not in df.columns:
        raise ValueError(f"parameter column '{args.param_field}' not found in data. available columns: {list(df.columns)}")
    
    if args.validate:
        max_param_value = df[args.param_field].max()
        max_param_models = df[df[args.param_field] == max_param_value]['model_size'].unique()
        
        if len(max_param_models) > 1:
            print(f"Warning: Multiple models have the same max parameter value {max_param_value:.1e}")
            print(f"  Models: {max_param_models}")
            print(f"  Using the first one: {max_param_models[0]}")
        
        val_model_size = max_param_models[0]
        
        df_val = df[df['model_size'] == val_model_size].copy()
        df_train = df[df['model_size'] != val_model_size].copy()
        
        print(f"\nData split (validation enabled):")
        print(f"  Training set: {len(df_train)} points (excluding {val_model_size})")
        print(f"  Validation set: {len(df_val)} points ({val_model_size} only, max params = {max_param_value:.1e})")
    else:
        df_train = df.copy()
        df_val = pd.DataFrame() 
        
        print(f"\nData split (validation disabled):")
        print(f"  Training set: {len(df_train)} points (all data)")
        print(f"  Validation set: 0 points")
    
    print("\nTraining set summary:")
    print(df_train[['model_size', 'token_multiplier', args.param_field, 'train_tokens', 'loss']])
    
    if args.validate and len(df_val) > 0:
        print("\nValidation set summary:")
        print(df_val[['model_size', 'token_multiplier', args.param_field, 'train_tokens', 'loss']])
    
    print(f"\nComputing fit quality metrics...")
    val_df = df_val if (args.validate and len(df_val) > 0) else None
    log_loss_for_ppl = args.longppl
    
    compute_fit_quality(
        df_train,
        val_df,
        epsilon=args.epsilon,
        param_field=args.param_field,
        formula_type=formula_type,
        loss_fn=args.loss_fn,
        log_loss_for_ppl=log_loss_for_ppl,
    )
    
    
    
if __name__ == "__main__":
    main()

