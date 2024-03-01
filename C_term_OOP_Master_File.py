import pandas as pd
import time
import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from C_term_1 import FilterUniprotFile, DownloadAlphaFoldFiles, Calculate_DSSP, TATMotifs
from C_term_2 import His6Center, IdentifyPlaceHolderResidues, COGs, FindCTerminus, SilhouetteScoresTYRs
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import re
from sklearn.utils import shuffle
import Bio.PDB.DSSP as DSSP
from Bio.PDB import PDBParser
import ast
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from Caddie_proteins import AnalyzeCaddieProteins

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

time_start = time.perf_counter()
# Search terms on Uniprot: Entry (=UniProt ID), Protein name, Organism, Taxonomic lineage, Seqeunce.
df_COs = pd.read_excel('C:\\Users\\panisf91-alt\\uniprotkb_catechol_oxidase_2024_01_26.xlsx')
df_TYRs = pd.read_excel('C:\\Users\\panisf91-alt\\uniprotkb_tyrosinase_2024_01_26.xlsx')

# Merged dfs containing TYRs and COs
df = pd.merge(df_TYRs, df_COs, how='outer', on=['Entry', 'Protein names', 'Sequence', 'Taxonomic lineage', 'Organism'])

df = df[df['Taxonomic lineage'].str.contains('Bacteria')]
print(f'length df after filtering for Bacteria: {len(df)}')

# Keep only entries that feature any of these substrings in their Protein name.
filter_string = 'tyrosinase|monophenol|polyphenol|catechol|uncharacterized|monooxygenase|melC2|1.10.3.1\)|1.14.18.1'

# Adds columns to the df: Mass, Kingdom, Phylum, Organism_clean, HxxxH.
# Filter df:
# 1. Kingdom = Archaea, or Bacteria, or Fungi, or Plant, or Metazoa
# 2. Sequence contains at least 8 histidins ---->>>>> 3.2.24: wurde deaktiviert!!!!!!!!!!!!!!
# 3. Sequence contains only amino acid letters
# 4. Protein name is in list_with_protein_names
# 5. Sequence has at least 1 HxxxH motif
# 6. Sequences from one Organism have less than identity_cutoff=0.98 (=98%) sequence identity
# stores resulting df as "os.path.join(os.getcwd(), f'filtered_df_{self.identity_cutoff}%_identity_{datetime.now().strftime("%d-%m-%Y")}.csv')"
x1 = FilterUniprotFile(df=df, filter_string=filter_string, identity_cutoff=0.98)

time_1st_filtering = time.perf_counter()
print(f'1st filtering took {round((time_1st_filtering - time_start) / 60,2)} minutes')

# Sets the pre-analyzed dataframe to df
df = x1.df
print(f'98% sequence identity df contains {len(df)} entries')

# Creates a new folder in the cwd called f'AlphaFoldFiles_{datetime.now().strftime("%d-%m-%Y")}' (e.g. AlphaFoldFiles_01.01.2024)
# Downloads all .pdb files of UniProt IDs present in the df available on AlphaFold and saves them in the newly created download folder
# If the internet connection is fast enough wait_n_second= can be reduced to 0.1
x2 = DownloadAlphaFoldFiles(df, wait_n_second=0.2)
x2.download_pdb_files()
time_downloading_files = time.perf_counter()
print(f'Downloading .pdb files took {round((time_downloading_files - time_1st_filtering) / 60,2)} minutes')

# Adds a column ('Has_pdb_file') to the df that contains 0 if no .pdb file could be retrieved from Alphafold for the corresponding entry and 1 if a .pdb file could be retrieved.
df = x2.check_pdb_file()

# Filters the df to contain only entries for which a .pdb file could be downloaded
df = df[df.Has_pdb_file == 1]

# creates a now column in df ('Has_6His_center') that contains a tuple that features at the following index positions::
# 0: True (if a 6 His center is present) / False (if no 6 His center is present)
# 1: The maximum distance of any of the six NE2 atoms present in the "best" 6 His center (lowest maximum distance).
# 2. The name of the .pdb file (format =AF-UniProtID-F1-model_v4.pdb)
# 3. The coordinates of the 6 histidins involved in the formation of the 6His center
df['classify_6His_center'] = df.UniProt_ID.apply(
    lambda x: His6Center().run_classification(UniProtID=x, pdb_folder_path=x2.download_folder))
df = df.copy()
df['Has_6His_center'] = df.classify_6His_center.apply(lambda x: x[0])
df['Max_dist_6His_center'] = df.classify_6His_center.apply(lambda x: x[1])
df['AF_file'] = df.classify_6His_center.apply(lambda x: x[2])
df['Coordinates_6_His'] = df.classify_6His_center.apply(lambda x: x[3])
# Removes sequences that do not contain a His6 center in their pdb file
df.drop(df[df.Coordinates_6_His.isna()].index, inplace=True)

# Saves and stores the dataframe
filename_2nd_time_filtered = os.path.join(os.getcwd(), f'2nd_time_filterd_df_{datetime.now().strftime("%d-%m-%Y")}.csv')
df.to_csv(filename_2nd_time_filtered, index=False)

# Loads the current dataframe. Load previously saved dataframe to start the analysis from here.
# df = pd.read_csv(filename_2nd_time_filtered, converters={'Coordinates_6_His': ast.literal_eval}) ##########################################################################################
print(f'6 His center checked df contains {len(df)} entries')

# Filters the df to contain only enzymes that feature a 6 His center in their AlphaFold structure (.pdb file).
# The column Has_6His_center contains a tuple ([0] = True/False). This code accesses the !st element of the tuple in column Has_6His_center
df = df[df.Has_6His_center == True]

time_2nf_filtering = time.perf_counter()
print(f'Checking 6 His centers took {round((time_2nf_filtering - time_downloading_files) / 60,2)} minutes')

# This stores the df in the cwd using the filename os.path.join(os.getcwd(), f'2nd_time_filterd_df_{datetime.now().strftime("%d-%m-%Y")}.csv')
# at this point the df contains the following modification:

# add columns to the df: Mass, Kingdom, Phylum, Organism_clean, HxxxH.
# filter df:
# 1. Kingdom = Archaea, or Bacteria, or Fungi, or Plant, or Metazoa
# 2. Sequence contains at least 8 histidins
# 3. Sequence contains only amino acid letters
# 4. Protein name is in list_with_protein_names
# 5. Sequence has at least 1 HxxxH motif
# 6. Sequences from one Organism have less than identity_cutoff=0.98 (=98%) sequence identity
# ---------------stored as f'1st_time_filtered_df_{self.identity_cutoff}%_identity_{datetime.now().strftime("%d-%m-%Y")}.csv') in the cwd-------------------------------------------
# 7. a .pdb file could be retrieved from AlphaFold for the entry
# 8. the pdb file retrieved for the entry features a 6 His center
# ---------------stored as f'2nd_time_filtered_df_{self.identity_cutoff}%_identity_{datetime.now().strftime("%d-%m-%Y")}.csv') in the cwd-------------------------------------------

# Removes Sequences in the MW fringe region below 70kDa that were manually checked and evaluated as invalid structures
drop_list = ['A0A2S6BQA0', 'M9M4Y3', 'A0A0D1CHR5', 'A0A132C9G1', 'A0A423VRG5', 'A0A0L8FPZ6', 'A0A139HFA6', 'A0A1H9LUS8',
             'A0A016ULR30', 'A0A409WEP5', 'A0A8B6GSA5', 'A0A7J6EUI5', 'A0A6M8NXE9', 'A0A1G5SEQ3', 'A0A1G5SEQ3',
             'A0A4V2AIA6', 'A0A2T5JQJ5',
             'A0A1I2YCN1', 'A0A127Z996', 'A0A0C3F7Q7', 'A0A7V9LZ51', 'A0A7C7I707', 'A0A7W0UL76', 'A0A2V6CJ46',
             'A0A1I4T9J4', 'A0A345T0B6',
             'A0A6J4QLW3', 'A0A7X9L660', 'A0A1K1MUG6', 'A0A7K3CNZ5', 'A0A1A9CU22', 'W2URW0', 'A0A0Q9IYK3', 'A0A285MR25', 'A0A2M9PD07',
             'A0A327RH85', 'A0A1I6VK36', 'A0A4Q6BC83', 'A0A2D0NGI4', 'A0A1V1ZHL4', 'E6X453', 'A0A0A7KCG3', 'A0A318IKS1', 'A0A1G7EVK2', 'A0A316L4G5',
             'K2PTI3', 'A0A6H1CAA7', 'A0A7V9HK32', 'A0A1I4RJF6', 'A0A2M9PDJ2', 'Q01WC9']
# Removes all Sequences with MW >= 70 kDa since these Structures are to a large extent erreneous.
df = df[df.Mass < 70000]
df.drop(df[df.UniProt_ID.isin(drop_list)].index, inplace=True)

# When running the whole scipr pdb_file_folder= should be x2.download_folder
# Calculates the fraction of AS involved in sheets and helices in the LAST nth fraction of the AS sequence

df['DSSP_C_term'] = df.apply(
    lambda x: Calculate_DSSP().execute_calculation(UniProt_ID_=x.UniProt_ID, Sequence=x.Sequence,
                                                   pdb_file_folder=x2.download_folder), axis=1)
df['DSSP_C_term_sheet'] = df.DSSP_C_term.apply(lambda x: x[0])
df['DSSP_C_term_helix'] = df.DSSP_C_term.apply(lambda x: x[1])
df['DSSP_sequence'] = df.DSSP_C_term.apply(lambda x: x[2])

# When running the whole scipr pdb_file_folder= should be x2.download_folder
# Calculates the fraction of AS involved in sheets and helices in the FIRST nth fraction of the AS sequence
calc_dssp_N_term = Calculate_DSSP()
df['DSSP_N_term'] = df.apply(
    lambda x: calc_dssp_N_term.execute_calculation(UniProt_ID_=x.UniProt_ID, Sequence=x.Sequence,
                                                   pdb_file_folder=x2.download_folder, C_term=False), axis=1)
df['DSSP_N_term_sheet'] = df.DSSP_N_term.apply(lambda x: x[0])
df['DSSP_N_term_helix'] = df.DSSP_N_term.apply(lambda x: x[1])

# stores the df as f'3rd_time_filterd_df_{datetime.now().strftime("%d-%m-%Y")}.csv'
# filter df:
# 1. Kingdom = Archaea, or Bacteria, or Fungi, or Plant, or Metazoa
# 2. Sequence contains at least 8 histidins
# 3. Sequence contains only amino acid letters
# 4. Protein name is in list_with_protein_names
# 5. Sequence has at least 1 HxxxH motif
# 6. Sequences from one Organism have less than identity_cutoff=0.98 (=98%) sequence identity
# ---------------stored as f'1st_time_filtered_df_{self.identity_cutoff}%_identity_{datetime.now().strftime("%d-%m-%Y")}.csv') in the cwd-------------------------------------------
# 7. a .pdb file could be retrieved from AlphaFold for the entry
# 8. the pdb file retrieved for the entry features a 6 His center
# ---------------stored as f'2nd_time_filtered_df_{self.identity_cutoff}%_identity_{datetime.now().strftime("%d-%m-%Y")}.csv') in the cwd-------------------------------------------
# 9. the fraction of AS involved in sheets and helices in the last nth fraction of the AS sequence has been calculated using DSSP

filename_3rd_time_filtered = os.path.join(os.getcwd(), f'3rd_time_filterd_df_{datetime.now().strftime("%d-%m-%Y")}.csv')
df = df[['UniProt_ID', 'Kingdom', 'Phylum', 'Organism_clean', 'Mass', 'DSSP_C_term_sheet', 'DSSP_C_term_helix',
         'Max_dist_6His_center', 'DSSP_N_term_sheet', 'DSSP_N_term_helix', 'DSSP_sequence',
         'AF_file', 'Coordinates_6_His', 'Protein_names', 'Sequence', 'Taxonomic_lineage']]
df.to_csv(filename_3rd_time_filtered, index=False)
time_calculate_DSSP = time.perf_counter()
print(f'Calculating DSSP took {round((time_calculate_DSSP - time_2nf_filtering) / 60,2)} minutes')
print(f'DSSP calculated df contains {len(df)} entries')

# loads the current dataframe. Load previously saved dataframe to start the analysis from here.
# currently correct file:'C:\\Anaconda\\Tools\\3rd_time_filterd_df_08-02-2024.csv'
# contains only bacteria, filtered for substrings in Protein_names
# df = pd.read_csv('C:\\Anaconda\\Tools\\3rd_time_filterd_df_12-02-2024.csv', converters={'Coordinates_6_His': ast.literal_eval})#################################################

# Manually classified bacterial TYRS. 0 = no C-terminus, 1 = C-terminus
classification_dict_bacteria = {'B5GKF1': 0, 'A0A542KAR7': 0, 'A0A423N8F1': 0, 'A0A1W6SPU4': 0, 'A0A856MD39': 0,
                                'A0A0T1T7G2': 1,
                                'A0A1V0UF35': 0, 'A0A1C4KL18': 0, 'A0A2U3H7T7': 0, 'A0A2V7VYD3': 1, 'A0A0N6W5U1': 0,
                                'A0A345RSB1': 0, 'A0A1M7KTF4': 1, 'A0A0B4Y5A7': 1, 'A0A1Q5CFP1': 0, 'A0A0N0LKI9': 1,
                                'A0A543VGD4': 0, 'A0A2P8QAZ3': 0, 'A0A5J6EZW4': 0, 'A0A429TFE4': 0,
                                'A0A534A679': 0, 'A0A1S2JWT4': 0, 'A0A3N1QGC9': 0, 'A0A0M9YN38': 0, 'A0A3D8NHV5': 0,
                                'A0A3M2LQ63': 1, 'A0A522AFU5': 1, 'A0A1L9B3D7': 0, 'A0A7Y7X987': 1, 'A0A1G5TK30': 0,
                                'A0A2D2AXZ0': 1, 'A0A367HXD7': 0, 'A0A318KMN6': 1, 'A0A1X2BDS2': 1, 'A0A6V8LZA0': 0,
                                'A0A4Z0L9R6': 0, 'A0A0F0HTL4': 0, 'A0A4R4C9L5': 1, 'A0A2N8TD21': 0, 'A0A4Q1FPC4': 1,
                                'A0A7W5X2N8': 1, 'A0A5E9PE47': 1, 'A0A101CKV0': 0, 'A0A447TAE7': 1, 'A0A372BP41': 1,
                                'A0A6I5N9R9': 0, 'K9TXE0': 1, 'A0A6B2RUU3': 0, 'A0A7W0WTN9': 0, 'A0A1V1ZHL4': 1,
                                'A0A7Y2MHT2': 0, 'A0A2S6Q084': 0, 'A0A0J6XJT5': 0, 'A0A2T5IGT7': 1, 'A0A2A2D8R7': 0,
                                'A0A2N8DQV3': 1, 'A0A1H8JR72': 1, 'A0A7W0GH75': 0, 'A0A209BXQ9': 0, 'A0A6I7YQ81': 0,
                                'A0A7Y7XW82': 1, 'A0A158KUU5': 1, 'A0A7Y0Z250': 1, 'A0A7V9ZIH2': 0, 'A0A212RWA1': 0,
                                'A0A4Q9RBJ6': 1, 'A0A0A3XV83': 1, 'A0A7X0Q103': 1, 'A0A359B6N4': 0, 'A0A5C6WRW5': 0,
                                'A0A3N5X831': 1, 'A0A101UNA9': 0, 'A0A0H4C1B8': 0, 'A0A160P4Z8': 0, 'A0A2S9E9Y7': 1,
                                'A0A2Z5JTE3': 0, 'A0A1H3FTE7': 1, 'A0A562MFH3': 1, 'A0A7G7XAD7': 1, 'A0A6G3TIK1': 0,
                                'A0A5B8Q256': 0, 'A0A7W7WLD5': 0, 'A0A0D4DDB6': 1, 'A0A4R3GYE8': 1, 'A0A646KQ41': 0,
                                'A0A1Q5JA69': 0, 'A0A4S5D3I6': 1, 'A0A235IC23': 0, 'A0A1V0QUI6': 0, 'A0A543VYF0': 0,
                                'A0A150X4H6': 1, 'A0A0F6L485': 1, 'A0A328BD42': 1, 'A0A0M8X0J6': 0, 'A0A7K2PFA8': 0,
                                'A0A840R7Q8': 1, 'A0A2T7KQJ4': 0, 'A0A1Q6XWM3': 0, 'A0A1H2IR23': 1, 'A0A2G0Z332': 1,
                                'A0A1Z4PY14': 0, 'A0A7Y5R7X6': 0, 'A0A7X0RVL1': 0, 'Q01WC9': 1, 'A0A3B8JCS8': 1,
                                'A0A7H0HS39': 0, 'A0A1D7MCL4': 1, 'A0A0C2PWS0': 1, 'A0A4Q3S7J8': 1, 'A0A3S1AKX4': 0,
                                'A0A7X0VVP0': 0, 'A0A841TEZ4': 0}

# Generates a Dataframe containing 'Mass', 'DSSP_sheet', 'DSSP_helix', 'Class' of all enzymes that were manually classified.
df['Class'] = df.UniProt_ID.apply(
    lambda x: classification_dict_bacteria[x] if x in classification_dict_bacteria.keys() else -1)
class_df = df[df.Class != -1][['Mass', 'DSSP_C_term_sheet', 'DSSP_C_term_helix', 'Class']]
class_df.index = df[df.Class != -1].UniProt_ID
class_df['Class'] = class_df.Class.astype('int')

# Scaling the data in 'Mass', 'DSSP_sheet', 'DSSP_helix'
scaler = MinMaxScaler()
X_train = scaler.fit_transform(class_df[['Mass', 'DSSP_C_term_sheet', 'DSSP_C_term_helix']])
y_train = np.array(class_df.Class)

# Data from the entire Dataframe that will be used to claclulate unknown classification
X_real = scaler.transform(df[['Mass', 'DSSP_C_term_sheet', 'DSSP_C_term_helix']])

# Generating and fitting model using manually classififed data
model_KNN = KNeighborsClassifier()
model_KNN.fit(X_train, y_train)

# Classifying unclassified enzymes.
pred = model_KNN.predict(X_real)
# Plots the Enzymes by DSSP_sheet and Mass and colours them according to the preictions amde by the model (KNN)
df_plot = pd.DataFrame(X_real, columns=['Mass', 'DSSP_C_term_sheet', 'DSSP_C_term_helix'], index=df.index)
df_plot['labels'] = pred
df_plot['ID'] = df.UniProt_ID

# 0 = no C-terminus, 1 = C-terminus. IF a manually determined label is available it is used, otherwise the computed label is assigned.
df['Pred_classes'] = pred
df['C_term'] = df.apply(lambda x: x.Pred_classes if x.Class == -1 else x.Class, axis=1)
phyla_with_at_least_ten_members = df.Phylum.value_counts()[df.Phylum.value_counts() >= 10].index.to_list()
df = df[df.Phylum.apply(lambda x: x in phyla_with_at_least_ten_members)]

"""fig = px.scatter(df, x='DSSP_C_term_sheet', y='Mass', color='C_term', hover_data={'UniProt_ID': True})
fig.update_traces(marker=dict(size=10))  # Adjust the marker size as needed
fig.update_layout(hoverlabel=dict(bgcolor="white", font_size=12), title=str(model))  # Adjust hover label appearance
fig.show()"""

df['Place_holder'] = df.apply(lambda x: IdentifyPlaceHolderResidues(
    pdb_file_path=os.path.join(x2.download_folder, x.AF_file),
    His6_center=x.Coordinates_6_His).identify_place_holder(), axis=1)
df['Place_holder_name'] = df.Place_holder.apply(lambda x: x[0])
df['Place_holder_ID'] = df.Place_holder.apply(lambda x: x[1])
df['Place_holder_distance'] = df.Place_holder.apply(lambda x: x[2])
df.drop(['Place_holder', ], axis=1)
"""# Calculates the distance between the COGs of the first 1/3 and and the last 1/3 of the AA sequence
df['COG_distance'] =df.AF_file.apply(lambda x: COGs(pdb_file_path=os.path.join('C:\Anaconda\Tools\AlphaFoldFiles_13-02-2024',x)).disatnce_C_term_N_term())
df['COG_distance'] = df.COG_distance.apply(lambda x: x[0])
# Calculates the Silhouette score between the first 1/3 and and the last 1/3 of the AA sequence
df['SC'] = df.AF_file.apply(lambda x: COGs(pdb_file_path=os.path.join('C:\Anaconda\Tools\AlphaFoldFiles_13-02-2024',x)).shilouette_score_C_N_term()"""
# A0A2M9PD07, A0A2M9PDJ2 -> C-terminus besteht aus C und N terminus

# Corrects wrong Place_holder residues
df.loc[df.UniProt_ID == 'A0A4Q6B2C2', 'Place_holder_name'] = 'LEU'
df.loc[df.UniProt_ID == 'A0A4Q6B2C2', 'Place_holder_ID'] = 437

# Determines the starting point of the C-terminal domain. Returns the IDs (in a pdb file) of all AAs that are present in the C-terminal domain and are involved in a beta sheet.
# The Minimum value corresponds to the starting point of the C-terminal domain
df['C_term_beta_sheets_IDs'] = df.apply(
    lambda x: FindCTerminus(pdb_file_folder=x2.download_folder, DSSP_sequence=x.DSSP_sequence,
                            UniProt_ID=x.UniProt_ID, C_term=x.C_term).run_analysis(), axis=1)

filename_4th_time_filtered = os.path.join(os.getcwd(), f'4th_time_filterd_df_{datetime.now().strftime("%d-%m-%Y")}.csv')
df.to_csv(filename_4th_time_filtered, index=False)

# Finds the ID (in a pdb file) of the amino acid that represents the start of the C-terminal domain

df['C_term_start'] = df.apply(lambda x: int(x.C_term_beta_sheets_IDs[0]) if x.C_term == 1 else -1, axis=1)
# Correts one enzyme, that has a wrongfully identified C-terminal domain
df.loc[df.UniProt_ID == 'A0A6M0RPD5', 'C_term_start'] = 463
df.loc[df.UniProt_ID == 'A0A7S7PZA0', 'C_term_start'] = 395
# Calculates which percentage of the AA seeunce forms the C-terminal domain
df['Perc_C_term'] = df.apply(lambda x: 100 - (x.C_term_start * 100 / len(x.Sequence)) if x.C_term == 1 else None, axis=1)
# Identifies the AA seqeunec of the C-terminal domain. The index of the starting AA is the ID - 1
df['C_term_sequence'] = df.apply(lambda x: x.Sequence[x.C_term_start - 1:] if x.C_term == 1 else None, axis=1)
# Calculates the mass of the C-terminal domain
df['C_term_mass'] = df.apply(lambda x: ProteinAnalysis(x.C_term_sequence).molecular_weight() if x.C_term == 1 else None,
                             axis=1)
# Calculates the mass of the active domain                             
df['Mass_active'] = df.apply(lambda x: ProteinAnalysis(x.Sequence[:x.C_term_start]).molecular_weight() if x.C_term == 1 else None, axis=1)
# Calculates the length of the C-terminal domain
df['C_term_length'] = df.apply(lambda x: len(x.C_term_sequence) if x.C_term == 1 else None, axis=1).fillna(-1).astype(
    int, errors='ignore')

# Checks which enzymes feature a CxxC motif within their C-terminal domain. Return True or False
df['CxxC_C_term'] = df.apply(
    lambda x: False if re.search(r'C[^C][^C]C', x.Sequence[x.C_term_start:]) == None else True, axis=1)

# Checks which enzymes feature a CxxC motif within their active domain. Return True or False
df['CxxC_C_term'] = df.apply(
    lambda x: False if re.search(r'C[^C][^C]C', x.Sequence[:x.C_term_start]) == None else True, axis=1)

df['SC_double_C_term'] = df.apply(lambda x: SilhouetteScoresTYRs(DSSP_sequeence=x.Sequence, UniProt_ID=x.UniProt_ID,
                                                    pdb_file_folder=x2.download_folder,
                                                    C_term_start=x.C_term_start).double_C_term() if x.C_term == 1 else None,
                     axis=1)

df['SC_C_term_active'] = df.apply(lambda x: SilhouetteScoresTYRs(DSSP_sequeence=x.Sequence, UniProt_ID=x.UniProt_ID,
                                                    pdb_file_folder=x2.download_folder,
                                                    C_term_start=x.C_term_start).SC_C_term_active() if x.C_term == 1 else None,
                     axis=1)  # replace pdb_file_folder with x2.download_folder########################

df['Count_beta_strands_temp'] = df.apply(lambda x: FindCTerminus().count_beta_strandes(
    seqeunce=x.DSSP_sequence, C_term_start=x.C_term_start) if x.C_term == 1 else None, axis=1)
df['Count_beta_strands'] = df.apply(lambda x: x.Count_beta_strands_temp[0] if x.C_term == 1 else None, axis=1)
df['List_beta_strands_IDs'] = df.apply(lambda x: x.Count_beta_strands_temp[1] if x.C_term == 1 else None, axis=1)
df['Length_beta_strands'] = df.apply(lambda x: [len(strand) for strand in x.List_beta_strands_IDs] if x.C_term == 1 else None, axis=1)
df.drop(['Count_beta_strands_temp', ], axis=1, inplace=True)


# Identifies structurs that haev a "double C-terminal domain" as all structures that have a SC_double_C_term value of > 0.41, except for few manually checked exceptions
df['Double_C_term'] = df.apply(lambda x: 1 if x.SC_double_C_term > 0.41 else 0, axis= 1)###############################change x.double_C_term to SC_double_C_term
double_C_term_exceptions = ['A0A7S7PZA0', 'A0A1I7MBS4', 'A0A1S9D310', 'G3A426']
df.loc[df.UniProt_ID.isin(double_C_term_exceptions), 'Double_C_term'] = 0

# The double C-terminal domain is NOT classified as a C-terminal domain, as it is not blocking access to the active site.
df.loc[df.Double_C_term == 1, 'C_term'] = 0

# Calculates the precentage of AAs in the C-terminal domain involved in beta_sheets
df['C_term_E'] = df.apply(
    lambda x: x.DSSP_sequence[x.C_term_start:].count('E') * 100 / len(x.DSSP_sequence[x.C_term_start:]), axis=1)
# Calculates the precentage of AAs in the C-terminal domain involved in alpha helices
df['C_term_H'] = df.apply(
    lambda x: x.DSSP_sequence[x.C_term_start:].count('H') * 100 / len(x.DSSP_sequence[x.C_term_start:]), axis=1)
# Calculates the precentage of AAs in the C-terminal domain involved in turns = loops
df['C_term_T'] = df.apply(
    lambda x: x.DSSP_sequence[x.C_term_start:].count('T') * 100 / len(x.DSSP_sequence[x.C_term_start:]), axis=1)

# Calculates the precentage of AAs in the active domain involved in beta_sheets
df['Active_E'] = df.apply(lambda x: (
        x.DSSP_sequence[:x.C_term_start].count('E') * 100 / len(
    x.DSSP_sequence[:x.C_term_start])) if x.C_term == 1 else (
        x.DSSP_sequence.count('E') * 100 / len(x.DSSP_sequence)), axis=1)

# Calculates the precentage of AAs in the active domain involved in alpha helices
df['Active_H'] = df.apply(lambda x: (
        x.DSSP_sequence[:x.C_term_start].count('H') * 100 / len(
    x.DSSP_sequence[:x.C_term_start])) if x.C_term == 1 else (
        x.DSSP_sequence.count('H') * 100 / len(x.DSSP_sequence)), axis=1)

# Calculates the precentage of AAs in the active domain involved in turns = loops
df['Active_T'] = df.apply(lambda x: (
        x.DSSP_sequence[:x.C_term_start].count('T') * 100 / len(
    x.DSSP_sequence[:x.C_term_start])) if x.C_term == 1 else (
        x.DSSP_sequence.count('T') * 100 / len(x.DSSP_sequence)), axis=1)
        
# Calculate the mass of the active domain      
df['Mass_active'] = df.apply(
    lambda x: ProteinAnalysis(x.Sequence[:x.C_term_start]).molecular_weight() if x.C_term == 1 else None, axis=1)

"""# Checks for the presence of a  TAT motif in teh C-terminal domain
df['TAT_C_term'] = df[df.C_term == 1].apply(
    lambda x: False if re.search(r'[ST]RR[A-Za-z][A-Za-z][LI]', x.Sequence[x.C_term_start:]) == None else True, axis=1)
df.loc[df.C_term == 0, 'TAT_C_term']= None

# Checks for the presence of a  TAT motif in the active domain
df['TAT_C_term'] = df.apply(
    lambda x: False if re.search(r'[ST]RR[A-Za-z][A-Za-z][LI]', x.Sequence[:x.C_term_start]) == None else True, axis=1)"""
    
# Check for the presence and starting AAs of TAT motifs in the seqeunces
TATs = TATMotifs(df=df)
df = TATs.df

# Save final TYR df
filename_5th_time_filtered = os.path.join(os.getcwd(), f'5th_time_filterd_df_{datetime.now().strftime("%d-%m-%Y")}.csv')
df.to_csv(filename_5th_time_filtered, index=False)
time_TYRs_finished = time.perf_counter()
print(f'Finishing TYr analysis took {round((time_TYRs_finished - time_start) / 60, 2)} minutes')

# Create dataframe with caddie proteins and analyze caddie proteins
# Provide path to xlsx file containing caddie proteins downloaded from the Uniprot Databank and path to the csv file containing completely filtered and analyzed TYRs.
c1 = AnalyzeCaddieProteins(df_caddie= x1.df_caddie, df_TYR_path=filename_5th_time_filtered)
time_finished = time.perf_counter()
print(f'All together took {round((time_finished - time_start) / 60, 2)} minutes')
print()
print()

####################################################################################
##==============================ANALYSIS STARTS HERE==============================##
####################################################################################
# Load dataframe containing filtered an analyzed TYRs (df)
# Load dataframe containing filtered an analyzed caddie proteins (df_caddie)
df = pd.read_csv(filename_5th_time_filtered) # replace path with "filename_5th_time_filtered" (without quotes) when running the full script.
df_caddie = pd.read_csv(c1.filename_df_caddie) # replace path with "c1.filename_df_caddie" (without quotes)when running the full script.

print('TYRs total:', len(df))
print('df.Phylum.value_counts():\n', df.Phylum.value_counts()[:2])
print('number of Streptomyces in total (with and without C-termoinal domain:',
      len(df[df.Taxonomic_lineage.str.contains('Streptomyces')]))

TYRs_with_C_term = len(df[df.C_term == 1])
print('_' * 30)
print('TYRs featuring a C-terminal domain:', TYRs_with_C_term)
print('% of TYRs featuring a C-terminal domain:', round(TYRs_with_C_term * 100 / len(df), 2), '%')
TYRs_without_C_term = len(df[df.C_term == 0])
print('TYRs featuring noa C-terminal domain:', TYRs_without_C_term)
print('% of TYRs featuring no C-terminal domain:', round(TYRs_without_C_term * 100 / len(df), 2), '%')
TYRs_no_C_term_from_Streptomyces = len(df[(df.C_term == 0) & (df.Taxonomic_lineage.str.contains('Streptomyces'))])
print('TYRs without a C-terminal domain produced by Streptomyces:', TYRs_no_C_term_from_Streptomyces)
TYRs_no_C_term_non_Streptomyces = len(df[(df.C_term == 0) & (~df.Taxonomic_lineage.str.contains('Streptomyces'))])
print('TYRs without a C-terminal domain NOT produced by Streptomyces:', TYRs_no_C_term_non_Streptomyces)
print('% of TYRs produced without C_term and Caddie', round(TYRs_no_C_term_non_Streptomyces * 100 / len(df), 2), '%')
print('permanently active TYRs', round(len(df[df.C_term == 0]) * 100 / len(df), 2), '%')
print('_' * 30)
df_value_counts_by_phylum = pd.DataFrame()
df_value_counts_by_phylum['total'] = df.value_counts('Phylum')
df_value_counts_by_phylum['C_term'] = df[df.C_term == 1].value_counts('Phylum')
df_value_counts_by_phylum['no_C_term'] = df[df.C_term == 0].value_counts('Phylum')
df_value_counts_by_phylum.fillna(0, inplace=True)
df_value_counts_by_phylum.loc['total'] = df_value_counts_by_phylum.sum(axis=0)
df_value_counts_by_phylum['%_C_term'] = round(
    df_value_counts_by_phylum['C_term'] * 100 / df_value_counts_by_phylum['total'], 2)
df_value_counts_by_phylum[['total', 'C_term', 'no_C_term']] = df_value_counts_by_phylum[
    ['total', 'C_term', 'no_C_term']].astype('int', errors='ignore')

print('number of ORganisms with/without C-terminal domai per Phylum\n', df_value_counts_by_phylum)
print('number of Streptomyces TYRs WITHa c-terminal domain: ',
      len(df[(df.Taxonomic_lineage.str.contains('Streptomyces')) & (df.C_term == 1)]))
print('UniProt_IDs of Streptomyces TYRs WITH  a C-terminal domain: \n',
      df[['UniProt_ID', 'Organism_clean']][(df.Taxonomic_lineage.str.contains('Streptomyces')) & (df.C_term == 1)])
print('\n')
#################################STRUCTUREOFTHEC-TERMINALDOMAIN###############################
print('Structure of the C-terminal domain'.upper())
min_C_term_mass = df.C_term_mass[df.C_term == 1].min()
max_C_term_mass = df.C_term_mass[df.C_term == 1].max()

print('lowest MW of a C-terminal domain:', round(min_C_term_mass / 1000, 2), 'kDa',
      f'({df.UniProt_ID[df.C_term_mass == min_C_term_mass].values[0]})')
print('highest MW of a C-terminal domain:', round(max_C_term_mass / 1000, 2), 'kDa',
      f'({df.UniProt_ID[df.C_term_mass == max_C_term_mass].values[0]})')

x = np.array(df[df.C_term == 1].Mass_active).reshape(-1, 1)
y = np.array(df[df.C_term == 1].C_term_mass).reshape(-1, 1)
reg = LinearRegression()
reg.fit(x, y)
pred = reg.predict(x)
print('R2 score of the correlation between the masses of the C-terminal domain and the active domain',
      round(r2_score(y, pred), 2))

print(f'Frequency of occurance of AA residues at the placeholder position',
      df[(df.C_term == 1) & (df.Place_holder_distance < 10)].value_counts('Place_holder_name', normalize=True) * 100)
TYRs_with_c_term_and_CxxC = len(df[(df.C_term == 1) & (df.CxxC_C_term == True)])
TYRs_without_c_term_and_CxxC = len(df[(df.C_term == 1) & (df.CxxC_C_term == False)])

print('TYRs with C_term WITH CxxC motif:', TYRs_with_c_term_and_CxxC, '=',
      round(TYRs_with_c_term_and_CxxC * 100 / len(df[df.C_term == 1]), 2), '%')
print('TYRs with C_term WITHOUT CxxC motif:', TYRs_without_c_term_and_CxxC, '=',
      round(TYRs_without_c_term_and_CxxC * 100 / len(df[df.C_term == 1]), 2), '%')

#################################ANALYSIS OF CADDIE PROTEINS STARTES HERE###############################
print()
print('analysis of caddie proteins starts here'.upper())
df_TYR = df.copy()

print('Number of investigated caddie proteins:', len(df_caddie))
print('Minimum mass of a caddie protein:', f'{round(df_caddie.Mass.min() / 1000, 2)} kDa',
      f'({df_caddie.UniProt_ID[df_caddie.Mass == df_caddie.Mass.min()].values[0]})')
print('Maximum mass of a caddie protein:', f'{round(df_caddie.Mass.max() / 1000, 2)} kDa',
      f'({df_caddie.UniProt_ID[df_caddie.Mass == df_caddie.Mass.max()].values[0]})')

#################################ANALYSIS OF TAT motifs STARTES HERE###############################
print()
print('analysis of TAT motifs starts here'.upper())
df_non_Strepto_TYR = df_TYR[~df_TYR.Taxonomic_lineage.str.contains('Streptomyces')]
print('Percentage (and absolut value) of Streptomyces Caddie proteins featuring a TAT motif:',
      f'{round(len(df_caddie[df_caddie.TAT == 1]) * 100 / len(df_caddie), 2)} % ({len(df_caddie[df_caddie.TAT == 1])})')
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

exit()
################################################################################################################
# --ENDE--ENDE--ENDE--ENDE--ENDE--ENDE--ENDE-ENDE--ENDE-ENDE--ENDE-ENDE--ENDE-ENDE--ENDE-ENDE--ENDE-ENDE--ENDE--#
################################################################################################################

protein_names_TYRs = ['Tyrosinase (EC 1.14.18.1) (Monophenol monooxygenase)', 'Tyrosinase', 'Tyrosinase family protein',
                      'Tyrosinase (EC 1.14.18.1)', 'Tyrosinase central domain-containing protein',
                      'Tyrosinase central domain protein', 'Tyrosinase copper-binding domain-containing protein']

protein_names_COs = ['catechol oxidase (EC 1.10.3.1)', 'catechol oxidase (EC 1.10.3.1) (Catechol oxidase)',
                     'Catechol oxidase (EC 1.10.3.1)', 'Polyphenol oxidase', 'Polyphenol oxidase (EC 1.10.3.1)',
                     'Polyphenol oxidase, chloroplastic (EC 1.10.3.1)',
                     'catechol oxidase (EC 1.10.3.1) (Catechol oxidase)',
                     'Polyphenol oxidase A1, chloroplastic (EC 1.10.3.1)', 'Polyphenol oxidase (EC 1.10.3.2)',
                     'Polyphenol oxidase (EC 1.14.18.1)', 'Polyphenol oxidase (EC 1.10.3.1, EC 1.14.18.1)']

# list with protein names (as defined in the UniProt DAtabank) that are included in the following investigations
list_with_protein_names = protein_names_COs + protein_names_TYRs

classification_dict_plants = {'A0A453JCH2': 1, 'A0A7I8JD17': 1, 'A0A803QVW1': 1, 'A0A7J8LZG0': 1, 'A0A6A3CU56': 1,
                              'A0A5D3DS75': 1, 'A0A7J9AIE8': 0, 'A0A4U6URN3': 1, 'A0A059CCV1': 1, 'A0A0N9DV50': 1,
                              'A0A2K1IZ55': 1, 'A0A6A6NKK1': 1, 'A0A2Z5DH90': 1, 'A0A1R3I1W8': 1, 'A0A2R6VZK5': 1,
                              'A0A2K1KIQ0': 1, 'A0A1B6Q6C0': 1, 'A0A498JYY9': 1, 'Q6YHK7': 1, 'A0A151T7G3': 1,
                              'A0A444ZWV1': 1, 'A0A7J8MVH9': 1, 'A0A2R6X3H8': 1, 'A0A118JXA6': 1, 'A0A2K1J0S4': 1,
                              'A0A2K1IGC8': 1, 'A0A498IS04': 1, 'A0A7D5L2D4': 1, 'A0A058ZW33': 1, 'Q948S3': 1,
                              'A0A6V7QWC5': 1, 'A0A445JYU3': 1, 'A0A068UD36': 1, 'A0A7I8JAS3': 1, 'M1BMR7': 1,
                              'A0A803NZB8': 1, 'A0A5D2TJG1': 1, 'O24056': 1, 'A0A445FXH7': 1, 'C7E3U6': 1,
                              'D8T5K2': 1, 'I3WE67': 1, 'A0A3B6CFK7': 1, 'A0A5N6NPH8': 1, 'A0A2U8ZU88': 1,
                              'A0A2J6KEG8': 1, 'A0A5D2GA62': 1, 'A0A7J9BSD6': 1, 'A0A5D2ULX1': 1, 'A9SJQ8': 1,
                              'A0A1U7XN75': 1, 'A0A2K1R853': 1, 'K3YGS6': 1, 'A0A803LPT1': 1, 'A0A7N2RCI0': 1,
                              'B0FIU2': 0, 'A5X3J8': 0, 'A0A7J7GY73': 0, 'A0A2R6X3H0': 1, 'A0A835DNL4': 1,
                              'A0A835ANI8': 1, 'A0A176W6H1': 1, 'A0A2G2XMF4': 1, 'A5X3P9': 1, 'A0A835V5C5': 1,
                              'A0A7J8QA71': 0, 'J3L4U1': 1, 'A0A7J8MXJ5': 1, 'A0A5D2Q517': 1, 'A0A0E0KVH0': 1,
                              'A0A2Z6NU82': 1, 'D7USP1': 1, 'M1BMR5': 1, 'F8V189': 1, 'A5X3L7': 1, 'A0A835MWN7': 1,
                              'B9VU65': 0, 'A0A2J6JY61': 1, 'I4DD55': 1, 'A0A0D4D5X8': 1, 'A0A5J9W460': 1,
                              'A0A7J9J501': 0, 'A0A7J7NY73': 1, 'A0A7I8L253': 1, 'A0A2J6LU56': 1, 'W9RL68': 1,
                              'A0A1S3YVF2': 1, 'A0A2J6KPC7': 1, 'V7CMC6': 1, 'K4CMI2': 1, 'A0A2J6JY65': 1,
                              'A0A142FJE7': 0, 'A0A811PN71': 1, 'A0A484K842': 1, 'A0A1Z5R8F8': 1, 'A0A1U7VZE2': 1,
                              'A0A1Y1HVY3': 1, 'A0A5A7QKU5': 1, 'A0A2P5YE33': 1, 'A0A1J6JLI9': 1, 'C0LU17': 1}

classification_dict_fungi = {'A0A3M7NYP3': 0, 'W6ZK40': 0, 'A0A4Q4XW69': 0, 'A0A4V1IWY8': 0, 'A0A0D2LDS6': 0,
                             'A0A1B8AVY5': 0, 'A0A5N5N5X4': 0, 'A0A0F7TKD9': 0, 'A0A1Z5TJY0': 0, 'A0A2V5HTW1': 0,
                             'A0A1R0GSB3': 0, 'A0A8A3PKD0': 1, 'A0A0C3D1N7': 0, 'A0A0F7S7B3': 0, 'A0A5N6JER4': 0,
                             'A0A447BTN0': 1, 'A0A0L0SML6': 0, 'A0A146FM76': 0, 'A0A1L9Q2X8': 0, 'A0A395SQA5': 0,
                             'A0A4Q9LG08': 0, 'Q92396': 1, 'A0A317WJT9': 0, 'A0A1Y2EBR8': 0, 'A0A4Q4NF14': 0,
                             'A0A443HRD0': 0, 'A0A439D4K9': 0, 'G1XM84': 0, 'A0A1Y2LZD4': 0, 'A0A1Y2URW8': 0,
                             'W7F796': 0, 'A0A5N6D8B2': 0, 'A0A395RSX2': 0, 'A0A2T9YAX1': 0, 'A0A093Y3Z5': 0,
                             'A0A439CXK6': 0, 'K3V5P2': 0, 'A0A135RS91': 0, 'A0A2P5HWP3': 0, 'A0A3L6NCJ7': 0,
                             'A0A3M2SNE1': 0, 'A0A0F7RVY0': 0, 'A0A5N6ZSA8': 0, 'A0A0C4EUF7': 0, 'S8ARG8': 0,
                             'A0A2G5HZD7': 0, 'A0A553HMY2': 0, 'S8AP44': 0, 'A0A7C8MYP4': 0, 'A0A4Z1HN92': 1,
                             'A0A3M7GA40': 0, 'X0LKR4': 0, 'A0A1B8GPV7': 0, 'A0A4V1IRJ7': 0, 'A0A3M7GQX7': 0,
                             'A0A4P9YS92': 0, 'A0A0F7ZS77': 0, 'A0A218Z5Z3': 0, 'A0A066Y1V3': 1, 'A0A4Q4QYU9': 0,
                             'A0A1Y1ZNT3': 0, 'A0A175VT61': 0, 'A0A093ZLM0': 0, 'A0A066W7C5': 0, 'A0A1G4BS67': 0,
                             'A0A1V6SI39': 0, 'A0A178CTE4': 0, 'A0A1Y6LZT0': 0, 'A0A409WQU7': 0, 'W9VMR6': 0,
                             'A0A1B8ARE9': 0, 'A0A3M7ALZ0': 0, 'A0A179HB00': 1, 'A0A0C2XG69': 0, 'A0A319DEV9': 0,
                             'A0A2T9ZGW2': 0, 'L7IVA7': 0, 'A0A4Q4ME11': 0, 'A0A401L4E0': 0, 'A0A2L2TR84': 1,
                             'A0A136ISR9': 0, 'A0A6P8BJH6': 0, 'A0A0D2IWH9': 0, 'A0A167C8C2': 0, 'A0A517LK50': 0,
                             'A0A2H1H3J1': 0, 'A0A2S6C090': 0, 'A0A420IKX8': 1, 'A0A4Q4XXZ9': 0, 'A0A7C8I6S9': 0,
                             'W7MUF6': 0, 'A0A084G593': 0, 'A0A4P7NS78': 0, 'A0A1J9R2U7': 0, 'A0A7U2ESQ2': 0,
                             'A0A4P9Z5J7': 0, 'A0A2T3AX63': 0, 'A0A6H0XRQ1': 0, 'A0A369H0I1': 1, 'A0A401L7K2': 0,
                             'Q2GNG2': 0, }
