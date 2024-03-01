import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio import pairwise2
import re
import os
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
import Bio.PDB.DSSP as DSSP
from Bio.PDB import PDBParser
from sklearn.utils import shuffle


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


class FilterUniprotFile:
    '''This class takes a pd.DataFrame (df) anf performes calculations anf filtering on the df. The input df represent a .xlsx file downloaded from the UniProt Databank
    with the features: Entry (=UniProt ID), Protein name, Organism, Taxonomic lineage, Seqeunce.

    The output df features the Kingdom, Phylum and Mass of each entry.
    Sequences containing non.amino acid letters will be removed.
    Also, sequences featuring less than n histidines (default. n=8) or no HxxxH motif will be removed.
    Also, sequences from one Organism that show a sequence identity above a certain threshold (default = 0.98) will be removed.'''

    def __init__(self, df, filter_string=None, identity_cutoff=0.98):
        self.identity_cutoff = identity_cutoff
        self.df = df
        if filter_string != None:
            self.filter_string = filter_string
            self.df.rename(
                columns={'Entry': 'UniProt_ID', 'Taxonomic lineage': 'Taxonomic_lineage', 'Protein names': 'Protein_names'},
                inplace=True)
            # Removes all entries that have a Protein name that does not represent a TYR
            self.df = self.df[self.df.Protein_names.str.contains(self.filter_string, case=False, regex=True)]
            # Removes all tyrosine-phenol-lyases from the df
            self.df = self.df[~self.df.Protein_names.str.contains('lyase')]
            # Removes as rows that contain non-amino acid letters in their amino acid sequence (e.g.'X')
            self.df = self.df[~self.df.Sequence.str.contains('B|J|O|U|X|Z', case=False, regex=True)]
            # Identifies the Kingdom from the Taxonomic_lineage
            self.df['Kingdom'] = self.df.Taxonomic_lineage.apply(lambda x: self.get_kingdom(x))
            # Only use bacterial TYRs for further analysis. Eucaryotic TYRs contain introns and are, therefore, truncated.
            self.df = self.df[self.df.Kingdom == 'Bacteria']
            # Identifies the Phylum from the Taxonomic_lineage
            self.df['Phylum'] = self.df.Taxonomic_lineage.apply(lambda x: self.get_phylum(x))
            # Identifies the latin Organisms name without species IDs from the Taxonomic_lineage
            self.df['Organism_clean'] = self.df.Organism.apply(lambda x: self.organism_clean(x))
            # Calculates the molecular mass of the Protein
            self.df['Mass'] = self.df.Sequence.apply(lambda x: ProteinAnalysis(x).molecular_weight())
            # Saves an alternative dataframe for investigating caddie proteins.
            self.df_caddie = self.df
            # Removes all seqeunces that feature less than 6 histidines in their structure
            self.df = self.df[self.df.Sequence.str.count('H') >= 6]
            # Checks if a HxxxH matif is present in the AA sequence
            self.df['HxxxH'] = self.df.Sequence.apply(lambda x: self.has_HxxxH(x))
            self.df = self.df[self.df.HxxxH == 1]
            # Dropping Enzymes that have Kingdom 'no-Kongdom-available' or 'Heunggongvirae'.
            self.df = self.df.drop(self.df.index[self.df.Kingdom.isin(['not_available', 'Heunggongvirae'])])
            # Remove 100% duplicates
            self.df = self.df.drop_duplicates(subset='Sequence', keep='first', inplace=False)
            # Removes sequences that are more similar than a predefined value (default = 98%)
            self.remove_similar_sequences()
            self.filename = os.path.join(os.getcwd(),
                                         f'1st_time_filtered_df_{self.identity_cutoff}%_identity_{datetime.now().strftime("%d-%m-%Y")}.csv')
            #self.df.to_csv(self.filename, index=False)

    def get_kingdom(self, lineage):
        '''Returns either Archaea, Bacteria, Plant, Fungi, or Metazoa as kingdom'''
        try:
            superkingdom = [i for i in lineage.split(',') if '(superkingdom)' in i]
            superkingdom_clean = superkingdom[0].replace('(superkingdom)', '').replace(' ', '')
            if 'Bacteria' in superkingdom_clean or 'Archaea' in superkingdom_clean:
                return superkingdom_clean
            else:
                kingdom = [i for i in lineage.split(',') if '(kingdom)' in i]
                kingdom_clean = kingdom[0].replace('(kingdom)', '').replace(' ', '')
                return kingdom_clean
        except IndexError:
            return 'not_available'

    def has_HxxxH(self, sequence):
        '''Rhecks for the presence of the HxxxH motif characteristic for TYRs and COs'''
        pattern = r'[^H][^H]H[^H][^H][^H]H[^H][^H]'
        if re.search(pattern, sequence):
            return 1
        else:
            return 0

    def get_phylum(self, string):
        '''Extracts the respective phylum out of the Taxonomic_lineage string downloaded from the UniProt Databank'''
        try:
            return [i for i in string.split(',') if 'phylum' in i][0].split(' ')[1]
        except IndexError:
            return 'not_available'

    def organism_clean(self, Organism):
        '''Adds a column that contains the latin Genus and Species name'''
        if ' (' in Organism:
            return Organism.split(' (')[0]
        else:
            return Organism

    def remove_similar_sequences(self):
        '''Removes sequences that originate from the same species and show more than n % identity. Only the first sequence present in the input file is retained'''
        self.drop_list_external = []
        for i in range(len(self.df.Organism.value_counts())):
            self.df_results = pd.DataFrame()
            self.df_preliminary = self.df[self.df.Organism == self.df.Organism.value_counts().index[i]]

            self.list1 = []
            for j in range(self.df_preliminary.Sequence.count()):
                self.drop_list_internal = []
                for k in range(self.df_preliminary.Sequence.count()):
                    try:
                        self.sequences = [self.df_preliminary.Sequence.iloc[j], self.df_preliminary.Sequence.iloc[k]]
                        self.alignment = pairwise2.align.globalxx(*self.sequences)
                        self.seq_identity = self.alignment[0].score / max([len(i) for i in self.sequences])
                        if self.seq_identity > self.identity_cutoff and self.seq_identity < 1:
                            self.drop_list_internal.append(self.df_preliminary.index[k])
                            self.drop_list_external.append(self.df_preliminary.index[k])
                    except IndexError:
                        pass

                self.df_preliminary = self.df_preliminary.drop(self.drop_list_internal)
        self.df.drop(self.drop_list_external, inplace=True)


class DownloadAlphaFoldFiles:
    '''Downloads pdb files from AlphaFold. The UniProt IDs of the respective enzymes are provided in the rows of a pd.DataFrame.
    A browser window is opened for downloading each .pdb file and closed after n seconds (default: n=4). Depending on the speed of the internet connection and the
    number of files that should be downloaded this time can be adjusted using wait_n_second=n.
    chromedriver.exe has to be provided (same Version as Chrome installed on the PC, can be downloaded from: https://sites.google.com/chromium.org/driver/). If chromedriver.exe
    is placed in teh current working directory no ath_to_chromedriver_exe= has to be provided. Otherwise, the full path to chromedriver.exe has to be provided'''

    def __init__(self, df=None, wait_n_second=0.2, path_to_chromedriver_exe='C:\\Anaconda\\Tools\\chromedriver.exe',
                 download_folder=None):
        self.df = df
        self.wait_n_second = wait_n_second
        self.path_to_chromedriver_exe = path_to_chromedriver_exe
        self.download_folder = download_folder
        if self.download_folder == None:
            # creates a new folder in the cwd called f'AlphaFoldFiles_{datetime.now().strftime("%d-%m-%Y")}' (e.g. AlphaFoldFiles_01.01.2024)
            try:
                self.download_folder = os.path.join(os.getcwd(),
                                                    f'AlphaFoldFiles_{datetime.now().strftime("%d-%m-%Y")}')
                os.makedirs(self.download_folder)
            except FileExistsError:
                pass

    def download_pdb_files(self):
        chrome_options = Options()
        chrome_options.add_experimental_option("prefs", {"download.default_directory": self.download_folder})

        service = Service(executable_path=self.path_to_chromedriver_exe)
        driver = webdriver.Chrome(service=service, options=chrome_options)
        time.sleep(2)

        for ID in self.df.UniProt_ID:
            driver.get(f'https://alphafold.ebi.ac.uk/files/AF-{ID}-F1-model_v4.pdb')
            time.sleep(self.wait_n_second)
        time.sleep(3)
        driver.quit()

    def check_pdb_file(self):
        self.df['Has_pdb_file'] = 0
        for file in os.listdir(self.download_folder):
            self.UniProt_ID = file.split('-')[1]
            self.df.loc[self.df['UniProt_ID'] == self.UniProt_ID, 'Has_pdb_file'] = 1
        return self.df


class Calculate_DSSP:
    def __init__(self):
        self.count = 0

    def calculate_DSSP_2(self, UniProt_ID_, Sequence, pdb_file_folder, C_term_fraction=0.3,
                         C_term=True):  # pdb_file_folder=x2.download_folder
        '''This function takes a UniProt_ID, and the path to a folder containing the corresponding .pdb file (file_name_style = default AlphaFold style)
        and return a list that features:
        [0]: % of AS involved in beta-sheets in the C_term_fraction of the total AS sequence
        [1] % of AS involved in alpha helices in the C_term_fraction of the total AS sequence.
        C_term_fraction = This is how many AS are used starting from the C-terminus to calculate DSSP parameters.
        (If the total length of the AS sequence is 300 AS and C_term_fraction=0.3 then the last 100 AS are used)'''
        # gerenerates the path to the pdb file corresponding to the UniProtID
        path_to_pdb_file = os.path.join(pdb_file_folder, f"AF-{UniProt_ID_}-F1-model_v4.pdb")
        parser = PDBParser()
        structure = parser.get_structure(UniProt_ID_, path_to_pdb_file)
        model = structure[0]
        dssp = DSSP(model, path_to_pdb_file)
        self.dssp_sequence = ''.join(pd.DataFrame(dssp)[2])
        # calculates the DSSP Parameters for the Enzyme. dssp_df contains in column [2] an 'E' if the AS is involved in a betha-sheet
        # and a 'H' if the AS is involved in an alpha-sheet (or  '-', 'B', 'G', 'I', 'T', 'S' if its neither involved in a beta-sheet nor alpha-helix)
        if C_term == True:
            dssp_df = pd.DataFrame(dssp)[(-round(len(str(Sequence)) * C_term_fraction)):]
        else:
            dssp_df = pd.DataFrame(dssp)[:(round(len(str(Sequence)) * C_term_fraction))]
        # returns a list.
        # The elemnt in position [0] is the percentage of AS involved in beta-sheets (=E) (in the last 1/3 of the total AS sequence).
        # The elemnt in position [1] is the percentage of AS involved in alpha-helices (=H) (in the last 1/3 of the total AS sequence).
        if 'E' in dssp_df[2].value_counts().keys():
            B_sheet = (dssp_df[2].value_counts() * 100 / dssp_df[2].value_counts().sum())['E']
        else:
            B_sheet = 0
        if 'H' in dssp_df[2].value_counts().keys():
            A_helix = (dssp_df[2].value_counts() * 100 / dssp_df[2].value_counts().sum())['H']
        else:
            A_helix = 0
        return [B_sheet, A_helix, self.dssp_sequence]

    def execute_calculation(self, UniProt_ID_, Sequence, pdb_file_folder, C_term_fraction=0.3, C_term=True):
        self.count += 1
        try:
            x = self.calculate_DSSP_2(UniProt_ID_, Sequence, pdb_file_folder, C_term_fraction=C_term_fraction,
                                      C_term=C_term)
            if self.count % 500 == 0:
                print(f'calculated DSSP for {self.count} files')
            return x
        except KeyError:
            print('KEY ERROR:')
            print(UniProt_ID_)
            x = self.calculate_DSSP_2(UniProt_ID_, Sequence, pdb_file_folder, C_term_fraction=C_term_fraction,
                                      C_term=C_term)
            return x

class TATMotifs:
    def __init__(self, df, pattern = r'[ST]RR[A-Za-z][A-Za-z][LI]'):
        self.df = df
        self.pattern = pattern
        self.TAT_C_term()
        self.TAT_active()
        self.TAT_C_term_start()
        self.TAT_active_start()
        self.df['TAT_total'] = self.df.apply(lambda x: True if x[['TAT_C_term', 'TAT_active']].any() else False, axis=1)

    def TAT_C_term(self):
        '''Finds all TAT motifs in the C-terminal domain'''
        self.df['TAT_C_term'] = self.df[self.df.C_term == 1].apply(
            lambda x: False if re.search(self.pattern, x.Sequence[x.C_term_start:]) == None else True,
            axis=1)
        self.df.loc[self.df.C_term == 0, 'TAT_C_term'] = None

    def TAT_active(self):
        '''Finds all TAT motifs in the active domain'''
        # Finds all TAT motifs in the active domain of seqeunces featuring a C-terminal domain
        # [:x.C_term_start+5] includes TYT motifs that would be split by the beginning of the C-terminal domain
        self.df['TAT_active'] = self.df[self.df.C_term == 1].apply(
            lambda x: False if re.search(self.pattern, x.Sequence[:x.C_term_start+5]) == None else True,
            axis=1)
        # Finds all TAT motifs in the active domain of seqeunces lacking a C-terminal domain
        self.df.loc[self.df[self.df.C_term == 0].index,'TAT_active'] = self.df[self.df.C_term == 0].apply(
            lambda x: False if re.search(self.pattern, x.Sequence) == None else True,
            axis=1)

    def TAT_indices(self, sequence, start=0, stop=None):
        if stop == None:
            stop = len(sequence)
        # Defines the pattern
        pattern = re.compile(self.pattern)
        # Defines the string to be seacrhed
        string = sequence[start:stop]
        # Stores the amino acid positions of the starting points of the TAT motif (as defined in a .pdb file)
        AA_positions = []

        # Starts searching for matches
        match = pattern.search(string)

        # Iterates over all matches
        while match:
            # Get the start index of the match
            start_index = match.start()
            # Appends the start index + 1 (=AA_position, indexing starts at 0, the AA_position at 1) to the list of indices
            AA_positions.append(start_index + 1 + start)
            # Finds the next match starting from the end of the current match
            match = pattern.search(string, match.end())

        # Returns a list with the starting AA of the found TAT motifs
        return AA_positions

    def TAT_C_term_start(self):
        self.df['TAT_C_term_start'] = self.df.apply(
            lambda x: self.TAT_indices(sequence=x.Sequence, start=x.C_term_start - 1),
            axis=1)

    def TAT_active_start(self):
        self.df['TAT_active_start'] = self.df.apply(
            lambda x: self.TAT_indices(sequence=x.Sequence, stop=x.C_term_start),
            axis=1)
