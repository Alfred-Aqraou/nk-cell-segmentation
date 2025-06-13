import argparse
import pandas as pd
import os

def format_latex_table(df_top10):
    latex = []
    latex.append("\\begin{table}[h!]")
    latex.append("\\centering")
    latex.append("\\caption{Insert Caption}")
    latex.append("\\label{tab:top10_...}")
    latex.append("\\resizebox{\\textwidth}{!}{")
    latex.append("\\begin{tabular}{ccccccccc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Batch Size} & \\textbf{Loss Type} & \\textbf{Encoder} & \\textbf{Freeze Encoder} & \\textbf{LR} & \\textbf{Optimizer} & \\textbf{Mean Val Loss} & \\textbf{Mean Val IoU (\%)} & \\textbf{Std Val IoU} \\\\")
    latex.append("\\midrule")

    for _, row in df_top10.iterrows():
        latex.append(f"{int(row['batch_size'])} & {row['loss_type']} & {row['encoder']} & {row['freeze_encoder']} & {row['lr']} & {row['optimizer']} & {row['mean_val_loss']:.2f} & {row['mean_val_iou']*100:.2f} & {row['std_val_iou']:.3f} \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("}")
    latex.append("\\end{table}")

    return "\n".join(latex)

def standardize_column_values(df):
    df['loss_type'] = df['loss_type'].str.lower().map({
        'iou': 'IoU',
        'dice': 'Dice',
        'combined': 'Combined'
    })

    df['encoder'] = df['encoder'].apply(lambda x: str(x).lower())

    # Replace specific full checkpoint paths with short labels
    replacements = {
        '/work3/s234843/02466-project/trained_models/old/dino/70queb8d/dino-70queb8d-ep=399.ckpt': 'DINO_{B}',
        '/work3/s234843/02466-project/trained_models/dino/fdnzv5c2/dino-fdnzv5c2-ep=399.ckpt': 'DINO_{BW}'
    }

    df['encoder'] = df['encoder'].replace(replacements)

    # Capitalize general encoders like resnet
    df['encoder'] = df['encoder'].str.replace('resnet', 'ResNet', regex=False)

    df['optimizer'] = df['optimizer'].str.lower().map({
        'adam': 'Adam',
        'sgd': 'SGD'
    })

    return df

def main():
    parser = argparse.ArgumentParser(description="Sort a CSV file by a given column.")
    parser.add_argument('--file', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--parameter', type=str, required=True, help='Column name to sort by')
    parser.add_argument('--top', type=int, required=True, help='Top N best')
    args = parser.parse_args()

    # Read the CSV file
    try:
        df = pd.read_csv(args.file)
    except FileNotFoundError:
        print(f"Error: File '{args.file}' not found.")
        return
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV file: {e}")
        return

    # Check if the column exists
    if args.parameter not in df.columns:
        print(f"Error: Column '{args.parameter}' not found in CSV.")
        print("Available columns:", ", ".join(df.columns))
        return

    # Standardize values before sorting or output
    df = standardize_column_values(df)

    # Sort the DataFrame by the specified column
    df_sorted = df.sort_values(by=args.parameter, ascending=False)

    # Create output filename
    base, ext = os.path.splitext(args.file)
    output_file = f"{base}_sorted_by_{args.parameter}{ext}"

    # Save the sorted DataFrame
    df_sorted.to_csv(output_file, index=False)
    print(f"Sorted CSV saved to '{output_file}'")

    # Generate and print LaTeX table for top 10 rows
    df_top10 = df_sorted.head(args.top)
    latex_table = format_latex_table(df_top10)
    print("\nLaTeX Table for Top 10 Rows:\n")
    print(latex_table)

if __name__ == "__main__":
    main()
