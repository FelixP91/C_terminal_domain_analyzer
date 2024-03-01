import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from C_term_1 import FilterUniprotFile, DownloadAlphaFoldFiles
import re
from datetime import datetime
import os
from C_term_1 import TATMotifs

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True, precision=2)


class AnalyzeCaddieProteins:
    '''Analyzes the primary seqeunces of caddie proteins provided by a .xlsx file downloaded from the UniProt Databank'''
    def __init__(self, df_caddie, df_TYR_path):
        self.df_caddie = df_caddie
        self.df_TYR = pd.read_csv(df_TYR_path)

        # Select all Streptomyces enzymes
        self.df_caddie = self.df_caddie[self.df_caddie.Taxonomic_lineage.str.contains('Streptomyces')]
        # Filtering datatframe
        x1 = FilterUniprotFile(df=self.df_caddie, identity_cutoff=0.55)
        x1.remove_similar_sequences()
        self.df_caddie = x1.df

        # Remove all proteins, that have been identified as TYRs
        self.df_caddie = self.df_caddie[~self.df_caddie.UniProt_ID.isin(self.df_TYR.UniProt_ID)]
        self.df_caddie = self.df_caddie[self.df_caddie.Organism_clean.isin(self.df_TYR.Organism_clean)]
        print('Make sure, that there are not more caddie proteins in df_caddie than Streptomyces TYRs in df_TYR by increasing/reducing identity_cutoff=xxx')
        print('Caddie proteins in df_caddie:\n', len(self.df_caddie[self.df_caddie.Organism_clean.isin(self.df_TYR.Organism_clean)]))
        print('Streptomyces TYRs in df_TYR (dataframe used to investigate TYRs):\n',
              len(self.df_TYR[self.df_TYR.Taxonomic_lineage.str.contains('Streptomyces')]))

        # Check if a CxxC motif is present in the caddie protein
        self.df_caddie['CxxC'] = self.df_caddie.apply(
            lambda x: False if re.search(r'C[^C][^C]C', x.Sequence) == None else True, axis=1)

        # Check if a TAT motif is present in the caddie protein
        self.df_caddie['TAT'] = self.df_caddie.apply(
            lambda x: False if re.search(r'[ST]RR[A-Za-z][A-Za-z][LI]', x.Sequence) == None else True, axis=1)

        # Remove all proteins that are outliers in terms of mass (as defined in a boxplot)
        q25 = np.quantile(np.array(self.df_caddie.Mass), 0.25)
        q75 = np.quantile(np.array(self.df_caddie.Mass), 0.75)
        iqr = q75-q25


        self.df_caddie = self.df_caddie[(self.df_caddie.Mass > (q25 - iqr*1.5)) & (self.df_caddie.Mass < (q75 + iqr*1.5))]
        self.filename_df_caddie = os.path.join(os.getcwd(), f'df_caddie_{datetime.now().strftime("%d-%m-%Y")}.csv')
        self.df_caddie.to_csv(self.filename_df_caddie, index=False)


if __name__ == "__main__":
    exit()
    df = pd.read_csv('/Tools/df_caddie_23-02-2024.csv')
    df_TYR = pd.read_csv('/Tools/5th_time_filterd_df_20-02-2024.csv')

    TAT_TYRs = TATMotifs(df=df_TYR)
    df_TYR = TAT_TYRs.df



    ####################################################################################
    ##==============================ANALYSIS STARTS HERE==============================##
    ####################################################################################
    print('ANALYSIS STARTS HERE')
    print('Number of investigated caddie proteins:', len(df))
    print('Minimum mass of a caddie protein:', f'{round(df.Mass.min() / 1000, 2)} kDa',
          f'({df.UniProt_ID[df.Mass == df.Mass.min()].values[0]})')
    print('Maximum mass of a caddie protein:', f'{round(df.Mass.max() / 1000, 2)} kDa',
          f'({df.UniProt_ID[df.Mass == df.Mass.max()].values[0]})')
    print()

    df_non_Strepto_TYR = df_TYR[~df_TYR.Taxonomic_lineage.str.contains('Streptomyces')]
    print('Percentage (and absolut value) of Streptomyces Caddie proteins featuring a TAT motif:',
          f'{round(len(df[df.TAT == 1]) * 100 / len(df), 2)} % ({len(df[df.TAT == 1])})')
    Streptomyces_TYRs_with_TAT = len(
        df_TYR[(df_TYR.TAT_active == 1) & (df_TYR.Taxonomic_lineage.str.contains('Streptomyces'))])
    Count_Streptomyces_TYRs = len(df_TYR[df_TYR.Taxonomic_lineage.str.contains('Streptomyces')])

    Count_non_Streptomyces_TYRs = len(df_TYR[df_TYR.Taxonomic_lineage.str.contains('Streptomyces') == False])
    Non_Streptomyces_TYRs_with_C_term_TAT = len(df_non_Strepto_TYR[df_non_Strepto_TYR.TAT_C_term == 1])
    Non_Streptomyces_TYRs_with_active_TAT = len(df_non_Strepto_TYR[df_non_Strepto_TYR.TAT_active == 1])
    print('Percentage (and absolut value) of Streptomyces TYRs featuring a TAT motif:',
          f'{round(Streptomyces_TYRs_with_TAT * 100 / Count_Streptomyces_TYRs, 2)} % ({Streptomyces_TYRs_with_TAT})')

    print('Precentage (and absolut value) of non-Streptomyces TYRs featuring a TAT motif:',
          f'{round((Non_Streptomyces_TYRs_with_C_term_TAT + Non_Streptomyces_TYRs_with_active_TAT) * 100 / Count_non_Streptomyces_TYRs, 2)} % ({Non_Streptomyces_TYRs_with_C_term_TAT + Non_Streptomyces_TYRs_with_active_TAT})')

    print('Percentage (and absolut value) of non_Streptomyces TAT motifs in the active domain',
          f'{round(Non_Streptomyces_TYRs_with_active_TAT * 100 / (Non_Streptomyces_TYRs_with_active_TAT + Non_Streptomyces_TYRs_with_C_term_TAT), 2)} % ({Non_Streptomyces_TYRs_with_active_TAT})')
    print('Percentage (and absolut value) of non_Streptomyces TAT motifs in the C_terminal domain',
          f'{round(Non_Streptomyces_TYRs_with_C_term_TAT * 100 / (Non_Streptomyces_TYRs_with_C_term_TAT + Non_Streptomyces_TYRs_with_active_TAT), 2)} % ({Non_Streptomyces_TYRs_with_C_term_TAT})')
    print('Percentage (and absolut value) of non-Streptomyces TYRs WITH a TAT motif that also feature a C-terminal domain',
          f'{round(len(df_non_Strepto_TYR[(df_non_Strepto_TYR.TAT_total == 1) & (df_non_Strepto_TYR.C_term == 1)]) * 100 / len(df_non_Strepto_TYR[df_non_Strepto_TYR.TAT_total == 1]), 2)} % ({len(df_non_Strepto_TYR[(df_non_Strepto_TYR.TAT_total == 1) & (df_non_Strepto_TYR.C_term == 1)])})')

