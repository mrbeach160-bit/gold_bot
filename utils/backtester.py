# utils/backtester.py

import pandas as pd
import numpy as np

def run_backtest(data, initial_capital, sl_pips, tp_pips):
    """
    Menjalankan simulasi trading (backtest) pada data historis.
    """
    capital = initial_capital
    point = 0.01  # Untuk XAU/USD
    
    positions = []
    equity_curve = []
    
    in_position = False
    position_type = None
    entry_price = 0
    stop_loss = 0
    take_profit = 0

    for i in range(len(data)):
        row = data.iloc[i]
        
        # Cek apakah posisi saat ini harus ditutup (kena SL/TP)
        if in_position:
            pnl = 0
            close_position = False
            if position_type == 'BUY':
                if row['low'] <= stop_loss:
                    pnl = (stop_loss - entry_price)
                    close_position = True
                elif row['high'] >= take_profit:
                    pnl = (take_profit - entry_price)
                    close_position = True
            elif position_type == 'SELL':
                if row['high'] >= stop_loss:
                    pnl = (entry_price - stop_loss)
                    close_position = True
                elif row['low'] <= take_profit:
                    pnl = (entry_price - take_profit)
                    close_position = True
            
            if close_position:
                # Asumsi lot size 1 untuk PnL dalam poin, nanti dikalikan lot
                positions.append({'exit_price': entry_price + pnl, 'pnl_points': pnl, 'type': position_type})
                capital += pnl # Ini hanya contoh, perhitungan modal riil lebih kompleks
                in_position = False

        # Cek sinyal untuk membuka posisi baru
        if not in_position and row['final_signal'] in ['BUY', 'SELL']:
            in_position = True
            position_type = row['final_signal']
            entry_price = row['close']
            
            if position_type == 'BUY':
                stop_loss = entry_price - (sl_pips * point)
                take_profit = entry_price + (tp_pips * point)
            else: # SELL
                stop_loss = entry_price + (sl_pips * point)
                take_profit = entry_price - (sl_pips * point)
        
        equity_curve.append(capital)

    # --- Kalkulasi Metrik Kinerja ---
    if not positions:
        return {
            "Total Profit/Loss": 0, "Win Rate (%)": 0, "Total Trades": 0,
            "Max Drawdown (%)": 0, "Profit Factor": 0
        }, pd.DataFrame({'Equity': [initial_capital]})

    results_df = pd.DataFrame(positions)
    total_trades = len(results_df)
    wins = results_df[results_df['pnl_points'] > 0]
    losses = results_df[results_df['pnl_points'] < 0]
    
    win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
    total_pnl = results_df['pnl_points'].sum()
    
    gross_profit = wins['pnl_points'].sum()
    gross_loss = abs(losses['pnl_points'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    equity_series = pd.Series(equity_curve)
    peak = equity_series.expanding(min_periods=1).max()
    drawdown = (equity_series - peak) / peak
    max_drawdown = abs(drawdown.min()) * 100

    metrics = {
        "Total Profit/Loss (points)": round(total_pnl, 2),
        "Win Rate (%)": round(win_rate, 2),
        "Total Trades": total_trades,
        "Max Drawdown (%)": round(max_drawdown, 2),
        "Profit Factor": round(profit_factor, 2)
    }
    
    return metrics, pd.DataFrame({'Equity': equity_curve})
