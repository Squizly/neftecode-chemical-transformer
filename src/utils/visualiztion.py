def plot_presentation_dashboard(history, best_epoch, model, val_loader, t_sc, device, seed_idx):
    model.eval()
    val_preds, val_trues = [], []
    with torch.no_grad():
        for b in val_loader:
            p, _ = model(b[0].to(device), b[1].to(device), b[2].to(device), b[3].to(device))
            val_preds.append(t_sc.inverse_transform(p.cpu().numpy()))
            val_trues.append(t_sc.inverse_transform(b[4].cpu().numpy()))
    val_preds, val_trues = np.concatenate(val_preds), np.concatenate(val_trues)
    
    # --- 1. РАСЧЕТ МЕТРИК И ВЫВОД ТАБЛИЦЫ ---
    summary_data = []
    names = ["Viscosity", "Oxidation"]
    for i in range(2):
        r2 = r2_score(val_trues[:, i], val_preds[:, i])
        mae = mean_absolute_error(val_trues[:, i], val_preds[:, i])
        mape = np.mean(np.abs((val_trues[:, i] - val_preds[:, i]) / (val_trues[:, i] + 1e-7))) * 100
        std_err = np.std(val_trues[:, i] - val_preds[:, i])
        summary_data.append([names[i], f"{r2:.4f}", f"{mae:.4f}", f"{mape:.2f}%", f"±{std_err:.4f}"])

    print(f"\n" + "═"*85)
    print(f" FINAL VALIDATION REPORT ".center(85, " "))
    print("═"*85)
    print(tabulate(summary_data, 
                   headers=["Target Variable", "R2 Score", "MAE", "Rel. Error (MAPE)", "Error Std Dev"], 
                   tablefmt="fancy_grid"))
    print("═"*85 + "\n")

    # --- 2. НАСТРОЙКА ВИЗУАЛА ---
    sns.set_style("white") 
    
    fig, ax = plt.subplots(figsize=(16, 7))
    fig.suptitle(f'Model Training Dynamics', 
                 fontsize=22, fontweight='black', y=0.96, color='#1a1a1a')
    
    epochs_range = range(1, len(history['train']) + 1)
    
    color_train = '#27ae60' 
    color_val = '#f39c12'   
    color_best = '#e74c3c'  
    
    # --- ПОСТРОЕНИЕ ГРАФИКОВ (Тонкие линии: lw=1.8) ---
    ln1, = ax.plot(epochs_range, history['train'], label='Train Loss', 
                   color=color_train, lw=3.5, zorder=3)
    ax.fill_between(epochs_range, history['train'], alpha=0.04, color=color_train)
    
    ln2, = ax.plot(epochs_range, history['val'], label='Validation Loss', 
                   color=color_val, lw=3.5, alpha=0.9, zorder=4)
    ax.fill_between(epochs_range, history['val'], alpha=0.04, color=color_val)
    
    # Линия лучшей эпохи
    ln3 = ax.axvline(best_epoch, color=color_best, linestyle='--', lw=1.5, alpha=0.8, 
                     label=f'Best Model (Epoch {best_epoch})')
    best_val_loss = history['val'][best_epoch-1]
    ax.scatter([best_epoch], [best_val_loss], color=color_best, s=80, zorder=5, edgecolors='white', lw=1.5)

    # --- НАСТРОЙКА ОСЕЙ И ИНТЕРВАЛОВ ---
    ax.set_ylabel('')
    ax.set_xlabel('Epochs', fontsize=12, fontweight='bold', color='#555555', labelpad=10)
    
    # Начинаем строго с 0.0 по оси Y и с 1 эпохи по X (без пустых отступов)
    ax.set_ylim(bottom=0.0)
    ax.set_xlim(left=1, right=len(history['train']))
    
    # Увеличиваем количество интервалов на оси Y (например, ~15 делений)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=20)) # На X тоже делаем аккуратно
    
    ax.tick_params(axis='y', colors='#444444', labelsize=11, width=1.5, length=5)
    ax.tick_params(axis='x', colors='#444444', labelsize=11, width=1.5, length=5)

    # Единая сетка
    ax.grid(True, color='#e0e6ed', linestyle='-', linewidth=1.0, alpha=0.8)
    
    # Убираем лишние рамки
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    
    # --- ЛЕГЕНДА СПРАВА СВЕРХУ ---
    lines = [ln1, ln2, ln3]
    labels = [l.get_label() for l in lines]
    
    # Добавляем полупрозрачный белый фон легенде, чтобы она не сливалась с линиями сетки/графика, 
    # если они вдруг пересекутся в правом углу
    leg = ax.legend(lines, labels, loc='upper right', fontsize=13) 
    
    for text, color in zip(leg.get_texts(), [color_train, color_val, color_best]):
        text.set_color(color)
        text.set_fontweight('bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()