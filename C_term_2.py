import pandas as pd
from Bio.PDB import PDBParser
import numpy as np
from itertools import combinations
import math
import os
import Bio.PDB.DSSP as DSSP
from Bio.PDB import PDBParser
from sklearn.metrics import silhouette_score

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


class His6Center:

    def get_NE2_coordinates(self):
        '''return a list with the coordinates of all NE2 atoms of histidine residues in the pdb file'''
        # Parse the structure from the PDB file
        self.parser = PDBParser(QUIET=True)
        self.structure = self.parser.get_structure("protein", self.pdb_file_path)
        self.NE2_coordinates = []
        self.NE2_residue_id_list = []
        # Iterate over all models in the structure
        for model in self.structure:
            # Iterate over all chains in the model
            for chain in model:
                # Iterate over all residues in the chain
                for residue in chain:
                    # Check if the residue is a histidine
                    if residue.get_resname() == "HIS":
                        # Iterate over all atoms in the residue
                        for atom in residue:
                            # search for all NE2 N atoms.
                            if atom.name == 'NE2':
                                self.NE2_coordinates.append(list(atom.get_coord()))
                                self.NE2_residue_id_list.append(residue.id[1])
        return [self.NE2_coordinates, self.NE2_residue_id_list]

    def find_HxxxH_motifs(self, ):
        '''Takes a list with the residue ids of all histidines in a protein and returns a list of tupls.
        Each tuple contains the residue ids of one HxxxH motif (notH-notH-H-notH-notH-notH-H-notH-notH-'''
        self.HxxxH_motifs = []
        for NE2_residue_id in self.NE2_residue_id_list:
            self.HxxxH = False
            if NE2_residue_id + 4 in self.NE2_residue_id_list:
                self.HxxxH = True
                for i in [-2, -1, 1, 2, 3, 5, 6]:
                    try:
                        if NE2_residue_id + i in self.NE2_residue_id_list:
                            self.HxxxH = False
                    except IndexError:
                        pass
            if self.HxxxH == True:
                self.HxxxH_motifs.append((NE2_residue_id, NE2_residue_id + 4))
        return self.HxxxH_motifs

    def create_6His_sets(self):
        '''creates a list with tuples. Each tuple contains the coordinates (a list with 3 floats) of 6 NE2 atoms of histidines.
        Two of the histidines are involved in the formation of the HxxxH motof, the other 4 histidines are systematiccaly permutated.'''
        self.list_with_6His_sets = []
        # gets the coordinates of all NE2 atoms of histidines and residue ids of the corresponding histidines.
        self.NE2_coordinates, self.NE2_residue_id_list = self.get_NE2_coordinates()
        # generates a list with n=number_of_HxxxH_motifes elements. Each element conatins the two residue ids of the two histidines involved in one HxxxH motifes as a list of tuples.
        self.HxxxH_motifs = self.find_HxxxH_motifs()
        # loops over all HxxxH mortived
        for HxxxH_motif in self.HxxxH_motifs:
            # generates from the ID (of the histidine residue) of the NE2 atom the index (position in the self.NE2_residue_id_list_).
            self.HxxH_indices = [self.NE2_residue_id_list.index(HxxxH_motif[0]),
                                 self.NE2_residue_id_list.index(HxxxH_motif[1])]
            # removes all NE2 atoms involved in the formation of the currently looped-over HxxxH motifs from the list of NE2 atoms
            self.NE2_coordinates_temp = [i for j, i in enumerate(self.NE2_coordinates) if j not in self.HxxH_indices]
            # adds all NE2 atoms involved in HxxxH motifs to each set of 6 histidines
            His2_set = (self.NE2_coordinates[self.HxxH_indices[0]], self.NE2_coordinates[self.HxxH_indices[1]])
            # generates all possible 4 histidine sets with the remaining NE2 atoms of histidines not involved in the formation of the HxxxH motif.
            for His4_set in combinations(self.NE2_coordinates_temp, 4):
                # adds the 2 histidines involved in the formation of the HxxxH motif to each 4 histidine set and stores this 6 histidine set in list_with_6His_sets.
                His6_set = tuple(list(His2_set) + list(His4_set))
                self.list_with_6His_sets.append(His6_set)
        return self.list_with_6His_sets

    def check_six_histidine_center(self):
        '''Checks if there is a 6 histidine center in the structure of the preotein. The 6 histiidne center has to be composed of 6 histidines that all have their NE2 atoms
        in a maximum distance of 5A from teh center of gravity of themselves. Moreover, two of the six histidine have to be featured in a HxxxH motif. IF such a 6 histidine center is found
        the method returns "True" (Bool), if not "False" (Bool), the maximim distance of any of the 6 histidines involved in the fomrmation of the 6His center to its center of gravity,
        the name of the pdb file for the corresponding UniProt_ID, and the coordinates of the 6 histidinies involved in the formation of the 6His center.'''
        self.list_with_6His_sets = self.create_6His_sets()
        # removes sequences that do not feature any HxxxH motif
        if len(self.list_with_6His_sets) < 1:
            return (False, 100, self.pdb_file, None)
        self.max_distances = [100]
        # loops over each set of 6 NE2 atoms from histidines
        for His6_set in self.list_with_6His_sets:
            distances_per_set = []
            # calculates the center of gravity for each set of 6 NE2 atoms from histidines
            center_of_gravity = [
                sum(coord[0] for coord in His6_set) / len(His6_set),
                sum(coord[1] for coord in His6_set) / len(His6_set),
                sum(coord[2] for coord in His6_set) / len(His6_set)
            ]
            # calculates the distance of each NE2 atom to the center of gravity of the corresponding set of 7 NE2 atoms
            for NE2_atom in His6_set:
                distance = math.sqrt(
                    (NE2_atom[0] - center_of_gravity[0]) ** 2 + (NE2_atom[1] - center_of_gravity[1]) ** 2 + (
                            NE2_atom[2] - center_of_gravity[2]) ** 2)
                distances_per_set.append(distance)
            # stores the maximum distance of any NE2 atoms to the center of gravity

            # checks if the currently calculaed 6His center is the most "tighty packed" = "best" 6 His center and if so stores the coordinates of the
            # 6 histidines involved as self.best_6_His_center

            if max(distances_per_set) < min(self.max_distances):
                self.best_6_His_center = His6_set
            self.max_distances.append(max(distances_per_set))
            # if the maximum distance of any set of 6 histidindes is smaller that 5A, the enzyme is classified as a TYR (True), otherwise it not (False).
        if min(self.max_distances) < self.max_distance_from_center:
            return (True, min(self.max_distances), self.pdb_file, self.best_6_His_center)
        else:
            return (False, min(self.max_distances), self.pdb_file, self.best_6_His_center)

    def run_classification(self, UniProtID, pdb_folder_path='C:\\Users\\panisf91\\AF_Files_C-term_31.1.24',
                           max_distance_from_center=5):
        '''Runs the whole classification for a selected protein. Call this to execute the classification!
        This method requires the UniProt_ID of the enzyme and the path to the folder (not the path to the .pdb file!) containing the corresponding .pdb file
        with the name-style generated by AlphaFold (=AF-{UniProtID}-F1-model_v4.pdb).'''
        # generates the name of the the AlphaFold .pdb file from the UniProt ID
        self.pdb_file = f'AF-{UniProtID}-F1-model_v4.pdb'
        self.pdb_folder_path = pdb_folder_path
        # defines the cut-off value for the minimum distance of any of the 6 NE2 atoms (from the 6 histidines) from the center of gravity (of all 6 NE2 atoms).
        # default = 5 Angstrom
        self.max_distance_from_center = max_distance_from_center
        # generates the path to the respective pdb file of the enzyme.
        self.pdb_file_path = os.path.join(self.pdb_folder_path, self.pdb_file)
        # runs the classification
        return self.check_six_histidine_center()


class IdentifyPlaceHolderResidues:
    '''This class identifies the place holder residue protruding from the C-terminal domain into the catalytic pocket.'''

    def __init__(self, pdb_file_path, His6_center):
        self.pdb_file_path = pdb_file_path
        self.His6_center = np.array(His6_center)

    def calculate_COG_His6_center(self):
        '''THis calculates the center of gravoty of the His6_center.'''
        self.COG_His6_center = self.His6_center.mean(axis=0)

    def identify_place_holder(self):
        '''This identifies the place_holder residue as the residue located in the last 30% of the AA sequence (= the C-terminus)
        which center of gravity is closest to the center of gravity of the His6 center'''
        # Calculates the center of gravity of the His6_center
        self.calculate_COG_His6_center()
        self.parser = PDBParser(QUIET=True)
        self.structure = self.parser.get_structure("protein", self.pdb_file_path)
        # Iterate over all models in the structure
        for model in self.structure:
            # Iterate over all chains in the model
            for chain in model:
                # Iterate over all residues in the chain
                self.length = 0
                # finds the length of the AA sequence
                for residue in chain:
                    if residue.id[1] > self.length:
                        self.length = residue.id[1]
                self.list_with_residue_COGs = []
                for residue in chain:
                    # Loops over the last 30 % of the AA sequence (= the C-terminus)
                    if residue.id[1] > (self.length * 0.7):
                        # Creates an empty np.arary
                        atoms_per_residue = np.empty((0, 3))
                        for atom in residue:
                            # Selects all atoms that are not backbone O,C, or N (but keeps the C_alpha)
                            if atom.name not in ['N', 'C', 'O']:
                                # Stores the coordinates of  all atoms that are not backbone O,C, or N (but keeps the C_alpha) in an np.array
                                atoms_per_residue = np.vstack([atoms_per_residue, atom.get_coord()])
                        # Calculates the center of gravity (COG) for that residue side chain (including the C_alpha)
                        COG = atoms_per_residue.mean(axis=0)
                        # Stores the COGs of all AAs in the C-terminus in list_with_COGs at inedt position [0] of a tuple
                        # with the ID of the respective residue at index position [1] and the residue_name at index position [2]
                        self.list_with_residue_COGs.append((COG, residue.id[1], residue.get_resname()))
        self.place_holder_distance = 100
        self.place_holder_ID = None
        # Iterates over all centers of gravity of residues located in the last 30% of the AA sequence (=the C-terminus)
        for element in self.list_with_residue_COGs:
            distance = math.sqrt((element[0][0] - self.COG_His6_center[0]) ** 2 + \
                                 (element[0][1] - self.COG_His6_center[1]) ** 2 + \
                                 (element[0][2] - self.COG_His6_center[2]) ** 2)
            # Selects the AA which ceter of gravity is closest to the center of gravity of the His6_center as the place_holder residue
            if distance < self.place_holder_distance:
                self.place_holder_distance = distance
                self.place_holder_ID = element[1]
                self.place_holder_name = element[2]
        # Return [0]: the name (3-letter code) of hte place_holder residue, [1]: its ID number, and [2]: the distance of its COG to the COG of the His6_center
        return (self.place_holder_name, self.place_holder_ID, self.place_holder_distance)


class COGs:
    '''This class calculates the center of gravities (COGs) of the C-terminus (last 30% of the AA sequence) and the N-terminus (first 30% of the AA sequence) '''

    def __init__(self, pdb_file_path, C_term_fraction=0.7, N_term_fraction=0.3):
        self.pdb_file_path = pdb_file_path
        self.C_term_fraction = C_term_fraction
        self.N_term_fraction = N_term_fraction
        self.CA_coordinates_C = None
        self.CA_coordinates_N = None

    def COG_C_terminus(self):
        '''Calculates the center of gravities (COGs) of the C-terminus (last 30% of the AA sequence)'''
        self.parser = PDBParser(QUIET=True)
        self.structure = self.parser.get_structure("protein", self.pdb_file_path)
        self.CA_coordinates_C = np.empty((0, 3))
        # Iterate over all models in the structure
        for model in self.structure:
            # Iterate over all chains in the model
            for chain in model:
                # Iterate over all residues in the chain
                self.length = 0
                # finds the length of the AA sequence
                for residue in chain:
                    if residue.id[1] > self.length:
                        self.length = residue.id[1]
                for residue in chain:
                    # Loops over the last 30 % of the AA sequence (= the C-terminus)
                    if residue.id[1] > (self.length * self.C_term_fraction):
                        for atom in residue:
                            # Selects all atoms that are not backbone O,C, or N (but keeps the C_alpha)
                            if atom.name == 'CA':
                                # Stores the coordinates of  all atoms that are not backbone O,C, or N (but keeps the C_alpha) in an np.array
                                self.CA_coordinates_C = np.vstack([self.CA_coordinates_C, atom.get_coord()])
        self.cog_C_term = self.CA_coordinates_C.mean(axis=0)

    def COG_N_terminus(self):
        '''Calculates the center of gravities (COGs) of the N-terminus (last 30% of the AA sequence)'''
        self.parser = PDBParser(QUIET=True)
        self.structure = self.parser.get_structure("protein", self.pdb_file_path)
        self.CA_coordinates_N = np.empty((0, 3))
        # Iterate over all models in the structure
        for model in self.structure:
            # Iterate over all chains in the model
            for chain in model:
                # Iterate over all residues in the chain
                self.length = 0
                # finds the length of the AA sequence
                for residue in chain:
                    if residue.id[1] > self.length:
                        self.length = residue.id[1]
                for residue in chain:
                    # Loops over the last 30 % of the AA sequence (= the C-terminus)
                    if residue.id[1] < (self.length * self.N_term_fraction):
                        for atom in residue:
                            # Selects all atoms that are not backbone O,C, or N (but keeps the C_alpha)
                            if atom.name == 'CA':
                                # Stores the coordinates of  all atoms that are not backbone O,C, or N (but keeps the C_alpha) in an np.array
                                self.CA_coordinates_N = np.vstack([self.CA_coordinates_N, atom.get_coord()])
        self.cog_N_term = self.CA_coordinates_N.mean(axis=0)

    def disatnce_C_term_N_term(self):
        '''Calculates the distance between the COGs of the C-terminus (last 30% of the AA sequence) and the N-terminus (first 30% of the AA sequence)'''
        self.COG_N_terminus()
        self.COG_C_terminus()
        self.distance = math.sqrt((self.cog_N_term[0] - self.cog_C_term[0]) ** 2 + \
                                  (self.cog_N_term[1] - self.cog_C_term[1]) ** 2 + \
                                  (self.cog_N_term[2] - self.cog_C_term[2]) ** 2)
        return (self.distance, self.cog_C_term, self.cog_N_term)

    def shilouette_score_C_N_term(self):
        '''Calculates the shilouette_score for CA atoms present in the C-terminus (last 30% of the AA sequence) and N_terminus (first 30% of the AA sequence)'''
        if self.CA_coordinates_C == None:
            self.COG_C_terminus()
        if self.CA_coordinates_N == None:
            self.COG_N_terminus()
        self.coordinates_array = np.vstack([self.CA_coordinates_C, self.CA_coordinates_N])
        self.labels_arary = np.hstack(
            [np.zeros(self.CA_coordinates_C.shape[0]), np.ones(self.CA_coordinates_N.shape[0])])
        self.silhouette_score_ = silhouette_score(self.coordinates_array, self.labels_arary)
        return self.silhouette_score_


class FindCTerminus:
    '''This class finds the starting point and ending point of the C-terminal domain. The ending point is the end of the AA sequence. The starting point is defined as the
    first AA involved in a beta-sheet that is spatially closer to the COG of the beta sheets in the C-terminal domain than it is to the COG of active domain'''

    def __init__(self, pdb_file_folder=None, UniProt_ID=None, DSSP_sequence=None, C_term=None):
        self.UniProt_ID = UniProt_ID
        self.pdb_file_folder = pdb_file_folder
        self.DSSP_sequence = DSSP_sequence
        self.C_term = C_term

    def find_all_beta_sheets(self):
        '''Finds the indices of all AAs that re involved in the formation of a beta strand with al least 2 AAs'''
        # Creates a list to store the indices a list
        self.indices_beta_strands = []
        index = 0
        # Iterates over all AAs (encoded as DDSp letter)
        while index + 1 <= len(self.DSSP_sequence):
            # Selects ass AA positions that are involved in a beta strand (=E)
            if self.DSSP_sequence[index] == 'E':
                consecutive_count = 0
                start_index = index
                # Counts the length of the current beta strand
                while (index + 1) <= len(self.DSSP_sequence) and self.DSSP_sequence[index] == 'E':
                    consecutive_count += 1
                    index += 1
                # Adds all AA IDs (as they are present in a pdb file) that re present in the current beta strand as a list
                if consecutive_count >= 2:
                    self.indices_beta_strands.append([start_index + 1 + i for i in range(consecutive_count)])
            index += 1

    def COG_beta_sheets(self):
        '''Finds the COG of the C-term as defined by the COG of the CAs of all AAs involved in the 5th last - 2nd last beta sheets'''
        if self.C_term != 1:
            return False
        # Selcts the last 6 beta sheets minus the very last (to exclude possibly misfolded outlier) that have at least 3 AAs
        self.pdb_file_path = os.path.join(self.pdb_file_folder, f"AF-{self.UniProt_ID}-F1-model_v4.pdb")
        self.CA_index_C_term = [i for i in self.indices_beta_strands if len(i) >= 3][-5:-1]
        # Flattens the list of lists
        self.CA_index_C_term = [item for sublist in self.CA_index_C_term for item in sublist]
        self.parser = PDBParser(QUIET=True)
        self.structure = self.parser.get_structure("protein", self.pdb_file_path)
        self.CA_coordinates_C_term = np.empty((0, 3))
        # Iterate over all models in the structure
        for model in self.structure:
            # Iterate over all chains in the model
            for chain in model:
                # Iterate over all residues in the chain
                for residue in chain:
                    # Loops over the last 30 % of the AA sequence (= the C-terminus)
                    if residue.id[1] in self.CA_index_C_term:
                        for atom in residue:
                            # Selects all atoms that are not backbone O,C, or N (but keeps the C_alpha)
                            if atom.name == 'CA':
                                # Stores the coordinates of all CA atoms that are present in the C-term domain in beta sheets
                                self.CA_coordinates_C_term = np.vstack([self.CA_coordinates_C_term, atom.get_coord()])
        # Calculates the COG
        self.COG_beta_sheets_ = self.CA_coordinates_C_term.mean(axis=0)

    def COG_active_domain(self):
        '''Finds the COG of the active domain by taking the COG of the CAs of the first 60 % of the AA sequence mius the first 10 %'''
        self.length = len(self.DSSP_sequence)
        # Selects the first 60 % of the AA sequence but excludes the very first 10 % (to remove incorresctly folded outliers)
        self.CA_index_active = [i for i in range(round(self.length * 0.1), round(self.length * 0.6))]
        self.pdb_file_path = os.path.join(self.pdb_file_folder, f"AF-{self.UniProt_ID}-F1-model_v4.pdb")
        self.parser = PDBParser(QUIET=True)
        self.structure = self.parser.get_structure("protein", self.pdb_file_path)
        self.CA_coordinates_active = np.empty((0, 3))
        # Iterate over all models in the structure
        for model in self.structure:
            # Iterate over all chains in the model
            for chain in model:
                # Iterate over all residues in the chain
                for residue in chain:
                    # Loops over the last 30 % of the AA sequence (= the C-terminus)
                    if residue.id[1] in self.CA_index_active:
                        for atom in residue:
                            # Selects all atoms that are not backbone O,C, or N (but keeps the C_alpha)
                            if atom.name == 'CA':
                                # Stores the coordinates of all CA atoms that are present in the active domain
                                self.CA_coordinates_active = np.vstack([self.CA_coordinates_active, atom.get_coord()])
        # Calculates the COG
        self.COG_active = self.CA_coordinates_active.mean(axis=0)

    def find_C_terminus(self):
        '''Dertermis the beginning of the C-terminal domain'''
        # Flattens the list with AAs involved in the beta sheets but keeps only beta sheets that are featured in the last 45 % of the AA sequence
        self.AAs_in_beta_sheets_index = [item for sublist in self.indices_beta_strands for item in sublist if
                                         sublist[0] > self.length * 0.563]
        # Creates an empty list to store the coordinates of CAs involved in beta sheets in the last 1/3 of the AA sequence
        self.CAs_in_beta_sheets_coords = []

        self.pdb_file_path = os.path.join(self.pdb_file_folder, f"AF-{self.UniProt_ID}-F1-model_v4.pdb")
        self.parser = PDBParser(QUIET=True)
        self.structure = self.parser.get_structure("protein", self.pdb_file_path)
        self.CA_coordinates_active = np.empty((0, 3))
        # Iterate over all models in the structure
        for model in self.structure:
            # Iterate over all chains in the model
            for chain in model:
                # Iterate over all residues in the chain
                for residue in chain:
                    # Loops over the last 30 % of the AA sequence (= the C-terminus)
                    if residue.id[1] in self.AAs_in_beta_sheets_index:
                        for atom in residue:
                            # Selects all atoms that are not backbone O,C, or N (but keeps the C_alpha)
                            if atom.name == 'CA':
                                # Stores the coordinates of  all CA atoms that are present in the last 1/3 of the AA sequence and are involved in a beta sheet.
                                self.CAs_in_beta_sheets_coords.append((list(atom.get_coord()), residue.id[1]))
        # Iterates over all CAs in the last 30% of the AA sequence which are involved in beta sheet formation and checks if they are closer to the COG of the C-term or to the
        # COG og the active domain. The first AA which is closer to the COG of the C-term is defined as the starting point of the C-terminal domain.
        self.start_C_term = []
        for CA, ID in self.CAs_in_beta_sheets_coords:
            distance_C_term = math.sqrt(
                (CA[0] - self.COG_beta_sheets_[0]) ** 2 + (CA[1] - self.COG_beta_sheets_[1]) ** 2 + (
                        CA[2] - self.COG_beta_sheets_[2]) ** 2)
            distance_active = math.sqrt(
                (CA[0] - self.COG_active[0]) ** 2 + (CA[1] - self.COG_active[1]) ** 2 + (
                        CA[2] - self.COG_active[2]) ** 2)
            if distance_C_term < distance_active:
                self.start_C_term.append(ID)
        return self.start_C_term

    def run_analysis(self):
        self.find_all_beta_sheets()
        if self.COG_beta_sheets() == False:
            return None
        self.COG_active_domain()
        result = self.find_C_terminus()
        return result

    def count_beta_strandes(self,seqeunce, min_length=3, C_term_start=0):
        '''Counts the number of beta strands in the C-terminal domian'''
        index = C_term_start - 1
        self.indices_beta_strands = []
        while index + 1 <= len(seqeunce):
            # Selects ass AA positions that are involved in a beta strand (=E)
            if seqeunce[index] == 'E':
                consecutive_count = 0
                start_index = index
                # Counts the length of the current beta strand
                while (index + 1) <= len(seqeunce) and seqeunce[index] == 'E':
                    consecutive_count += 1
                    index += 1
                # Adds all AA IDs (as they are present in a pdb file) that re present in the current beta strand as a list
                if consecutive_count >= min_length:
                    self.indices_beta_strands.append([start_index + 1 + i for i in range(consecutive_count)])
            index += 1
        return (len(self.indices_beta_strands), self.indices_beta_strands)


class SilhouetteScoresTYRs:
    def __init__(self, DSSP_sequeence, UniProt_ID, pdb_file_folder, C_term_start):
        self.DSSP_sequence = DSSP_sequeence
        self.UniProt_ID = UniProt_ID
        self.pdb_file_folder = pdb_file_folder
        self.pdb_file_path = os.path.join(self.pdb_file_folder, f"AF-{self.UniProt_ID}-F1-model_v4.pdb")
        self.C_term_start = C_term_start
        self.C_term_length = len(self.DSSP_sequence) - self.C_term_start

    def double_C_term(self):
        '''Calculates the SC of the first 30% of the C-terminal domain and the last 30% of the C-terminal domain.
        Used to identify TYRs with a "double C-terminal domain"
        High SC (>0.42) characterizes TYRs with a "double C-terminal domain". Exception = A0A1I7MBS4'''
        # Creates numpy arrays to store the coordinates of the CA atoms of the first and last 30% of the C-terminal domain
        Atoms_C_term_start_coords = []
        Atoms_C_term_end_coords = []
        # Creates lists storing the IDs of the CA atoms of the first and last 30% of the C-terminal domain
        Atoms_C_term_start_IDs = list(range(self.C_term_start, self.C_term_start + round(self.C_term_length * 0.3)))
        Atoms_C_term_end_IDs = list(
            range(round(self.C_term_start + self.C_term_length * 0.7), len(self.DSSP_sequence)))

        # Creates a list to store the labels (C_term_start = 0, C_term_ed = 1)
        labels = []
        label = 0
        # Loops over all AAs and stores their coordinates if
        for IDs, coords in ([Atoms_C_term_start_IDs, Atoms_C_term_start_coords],
                            [Atoms_C_term_end_IDs, Atoms_C_term_end_coords]):
            self.parser = PDBParser(QUIET=True)
            self.structure = self.parser.get_structure("protein", self.pdb_file_path)
            # Iterate over all models in the structure
            for model in self.structure:
                # Iterate over all chains in the model
                for chain in model:
                    # Iterate over all residues in the chain
                    for residue in chain:
                        # Loops over the last 30 % of the AA sequence (= the C-terminus)
                        if residue.id[1] in IDs and self.DSSP_sequence[residue.id[1] - 1 == 'E']:
                            for atom in residue:
                                # Stores the coordinates of  all CA atoms that are present in the last 1/3 of the AA sequence and are involved in a beta sheet.
                                coords.append(list(atom.get_coord()))
                                labels.append(label)
            label += 1
        self.SC1 = silhouette_score(X=np.vstack((Atoms_C_term_start_coords, Atoms_C_term_end_coords)), labels=np.array(labels))
        return self.SC1

    def SC_C_term_active(self):
        '''Calculates the SC of the first 30% of the C-teminal domain and the active domain.
        Used to identify C-termini that have been incorrectly identified and partially feature the active domain
        Low SC characterizes incorrectly identifies TYRs'''
        Atoms_C_term_start_coords = []
        Atoms_active_coords = []
        # Creates lists storing the IDs of the CA atoms of the first 30% of the C-terminal domain and the first 20-60% of the active domain
        Atoms_C_term_start_IDs = list(range(self.C_term_start, self.C_term_start + round(self.C_term_length * 0.3)))
        Atoms_active_IDs = list(
            range(round(self.C_term_start*0.2), round(self.C_term_start*0.6)))

        # Creates a list to store the labels (C_term_start = 0, C_term_ed = 1)
        labels = []
        label = 0
        # Loops over all AAs and stores their coordinates if
        for IDs, coords in ([Atoms_C_term_start_IDs, Atoms_C_term_start_coords],
                            [Atoms_active_IDs, Atoms_active_coords]):
            self.parser = PDBParser(QUIET=True)
            self.structure = self.parser.get_structure("protein", self.pdb_file_path)
            # Iterate over all models in the structure
            for model in self.structure:
                # Iterate over all chains in the model
                for chain in model:
                    # Iterate over all residues in the chain
                    for residue in chain:
                        # Loops over the last 30 % of the AA sequence (= the C-terminus)
                        if label != 0:
                            if residue.id[1] in IDs and self.DSSP_sequence[residue.id[1] - 1 == 'H']:
                                for atom in residue:
                                    if atom.name =='CA':
                                        # Stores the coordinates of all CA atoms in the selectes pasrt of the active domain that are involved in alpha helices
                                        coords.append(list(atom.get_coord()))
                                        labels.append(label)
                        else:
                            if residue.id[1] in IDs:
                                for atom in residue:
                                    if atom.name =='CA':
                                        # Stores the coordinates of all CA atoms that are present in the last 1/3 of the AA sequence
                                        coords.append(list(atom.get_coord()))
                                        labels.append(label)

            label += 1
        self.SC2 = silhouette_score(X=np.vstack((Atoms_C_term_start_coords, Atoms_active_coords)),
                                    labels=np.array(labels))
        return self.SC2

