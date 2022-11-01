import os
import argparse
import glob
import shutil
from itertools import combinations

import os
import json
from pathlib import Path

VALID_T1_EXTS = ['.nii', '.nii.gz']
VALID_DWI_EXTS = ['.bval', '.bvec', '.nii.gz']

FILE_DR = Path(__file__).parent.absolute()

def load_config():

    # Load some values from global config json
    with open(os.path.join(FILE_DR, 'config.json'), 'r') as f:
        config = json.load(f)

    return config

def load_submit_template():

    # Load from submit template
    with open(os.path.join(FILE_DR, 'submit_template.sh'), 'r') as f:
        template = f.readlines()

    return template

def _cl_input(prompt):

    user_input = input(prompt)
    return user_input.lower().rstrip().replace('"', '').replace("'", "")

def cl_input(prompt, valid=None):
    '''Helper method to clean input'''

    user_input = _cl_input(prompt)

    if valid is None:
        return user_input

    if valid == int:
        try:
            return int(user_input)
        except ValueError:
            print('Input must be integer!')
            return cl_input(prompt, valid=int)

    while user_input not in valid:
        print(f'Input must be one of {valid}!')
        user_input = _cl_input(prompt)

    return user_input

def init_pq_dr(dataset_dr):

    pq_dr = os.path.join(dataset_dr, 'derivatives', 'prequal')

    if os.path.exists(pq_dr):
        n_files = len(os.listdir(pq_dr))

        if n_files > 0:
            print(f'Existing prequal derivatives directory found, with {n_files} existing subject(s)/folder(s).')
            prompt = 'Type delete to deleting any existing files and subject folders (or press enter to do nothing, overwriting any existing subjects): '
            prompt = cl_input(prompt, valid=['', 'delete'])
            if prompt == 'delete':
                shutil.rmtree(pq_dr)

    os.makedirs(pq_dr, exist_ok=True)
    return pq_dr

def get_subjs_from_key(dataset_dr, sub_key='sub-*/dwi'):

    # Glob for dr's
    subject_dwi_drs = glob.glob(os.path.join(dataset_dr, sub_key))

    # Remove any with no files in dwi
    for s_dr in subject_dwi_drs.copy():
        if len(os.listdir(s_dr)) == 0:
            subject_dwi_drs.remove(s_dr)

    return subject_dwi_drs


def get_subjects(dataset_dr, dataset_name):

    # Single session default
    sub_key = 'sub-*/dwi'

    # Check for multi / single session
    sessions = glob.glob(os.path.join(dataset_dr, 'sub-*', 'ses-*'))
    if len(sessions) > 0:
        sub_key = 'sub-*/ses-*/dwi'

    # Try to get keys
    subject_dwi_drs = get_subjs_from_key(dataset_dr, sub_key=sub_key)

    # Loop until valid key passed
    while len(subject_dwi_drs) == 0:
        print(f'No valid subject dwi directories found with glob str: "{sub_key}".')
        print('Please enter a custom glob key, where * refers to wildcard, and ' \
              'any entered path with be applied after the dataset location.')
        print('Note: If using nested session file structure, make sure '\
              'that nested folders follow naming "dataset/sub-*/ses-*/dwi/"!')

        # Get user input and clean for str quotes
        sub_key = input('Custom subject directory glob key: ')
        sub_key = sub_key.replace('"', '').replace("'", "")

        # Try again w/ custom keep
        subject_dwi_drs = get_subjs_from_key(dataset_dr, sub_key=sub_key)

    # If session-style, set subject as sub_ses
    if 'ses-*' in sub_key:
        subject_names = ['_'.join(subj.split('/')[-3:-1]) for subj in subject_dwi_drs]

        n_subjects = len({name.split('_')[0] for name in subject_names})
        n_events = len({name.split('_')[1] for name in subject_names})

        if n_events == 1:
            print(f'Found {len(subject_names)} potential subjects in '\
                  f'dataset={dataset_name}, with dwi data.')
        else:
            print(f'Found {n_subjects} potential subjects with {n_events} unique '\
                  f'events in dataset={dataset_name}, that have dwi data.')

    # Single case
    else:
        subject_names = [subj.split('/')[-2] for subj in subject_dwi_drs]

        print(f'Found {len(subject_names)} potential subjects in '\
              f'dataset={dataset_name}, with dwi data.')

    print()
    return subject_dwi_drs, subject_names

def file_to_piece(file, name):
    '''Convert file name to piece name'''

    # Must start with subject name, if session style
    # then this includes session.
    if not file.startswith(name):
        return None

    # Don't include .json in piece
    if file.endswith('.json'):
        return None

    file = file.replace(name, '')
    piece = file.split('.')[0]
    return piece

def is_dwi_valid(piece, name, files):
    return all([name + piece + ext in files for ext in VALID_DWI_EXTS])

def is_t1_valid(piece, name, files):
    return any([name + piece + ext in files for ext in VALID_T1_EXTS])

def get_u_pieces(s_dr, name, check='dwi'):

    dwi_files = os.listdir(s_dr)
    pieces = set([file_to_piece(file, name) for file in dwi_files])

    # Remove any None's
    if None in pieces:
        pieces.remove(None)

    # Perform type specific checks on pieces
    if check == 'dwi':
        check_func = is_dwi_valid
    elif check == 't1':
        check_func = is_t1_valid
    else:
        raise RuntimeError()

    valid_pieces = set()
    for piece in pieces:
        if check_func(piece, name, dwi_files):
            valid_pieces.add(piece)
    return valid_pieces

def check_known(known, c1, c2):
    '''Known is dict from choice to set
    and also symmetrically, so only
    need to check from c1 to c2.'''

    def check_c2(conflicts):

        # Multiple c2's
        if isinstance(c2, tuple):

            # If any a conflict, then conflict
            for c in c2:
                if c in conflicts:
                    return True

            # Otherwise, not
            return False

        # c2 is single option
        else:
            return c2 in conflicts
 
    # Check multiple c1 case
    if isinstance(c1, tuple):
        for c in c1:
            if c in known:
                
                # If conflict, return / end
                if check_c2(known[c]):
                    return True

    # Single c1 case        
    else:
        if c1 in known:
            return check_c2(known[c1])
        
    # If here, then return False, no conflict
    return False

def add_to_known(known, c1, c2):
    '''Call this method twice, with order reversed of c1, c2.
    known is dict from single str choices to set of single str choices.'''
    
    if isinstance(c1, tuple):
        for c in c1:
            known = add_to_known(known, c, c2)
        return known
    
    if isinstance(c2, tuple):
        for c in c2:
            known = add_to_known(known, c1, c)
        return known
        
    # Base add case
    try:
        known[c1].add(c2)
    except KeyError:
        known[c1] = set([c2])
    
    return known

def new_piece(c1, c2):
    '''Combine either single strs, or mix of str tuples
    into one tuple representing combined piece.'''
    
    new = []
    for c in [c1, c2]:
        if isinstance(c, tuple):
            new += list(c)
        else:
            new.append(c)

    return tuple(new)

def get_incomplete(piece_info, n):
    '''Get list of incomplete.'''

    # Start off w/ init dict of incomplete
    incomplete = []
    for piece in piece_info:
        if len(piece_info[piece]) != n:
            incomplete.append(piece)

    return incomplete

def check_combos(piece_info, n, known=None):
    
    # This is dict of any previous user prompts
    if known is None:
        known = {}

    # Start off w/ dict of incomplete stubs
    incomplete = get_incomplete(piece_info, n)

    # Go through all possible combinations
    for combo in combinations(incomplete, 2):
        
        # Unpack combo
        c1, c2 = combo
        
        # Skip if known not a match, i.e., avoid
        # user re-prompts
        if check_known(known, c1, c2):
            continue
    
        # Check for any intersecting subjects, if any, skip this combo
        c1_subjs, c2_subjs = piece_info[c1], piece_info[c2]
        any_intersect = c1_subjs.intersection(c2_subjs)
        if len(any_intersect) > 0:
            continue
            
        # Otherwise, check with user to see if these should be
        # treated the same
        p = f'Should file stubs "{c1}" and "{c2}" be treated as the same? (y/n): '
        user_choice = cl_input(p, valid=['y', 'n', 'yes', 'no'])
            
        # Yes case, merge pieces into one tuple
        if user_choice == 'y' or user_choice == 'yes':
            
            # Remove old pieces and add new w/ union
            del piece_info[c1]
            del piece_info[c2]

            piece_info[new_piece(c1, c2)] = c1_subjs.union(c2_subjs)
            
            # Return after merged, w/ flag for not done
            return piece_info, known, False
            
        # No case, add answer to known
        else:
            
            # Add symetrically to known
            known = add_to_known(known, c1, c2)
            known = add_to_known(known, c2, c1)
    
    # Final return, with True flag for done
    return piece_info, known, True

def _incomplete_prompt(incomplete, piece_info, n):

    print('Consider the following options:')
    print('- Input enter (an empty string) to continue as is.')
    print(f'- Input "delete" to remove all {len(incomplete)} incomplete file stubs.')

    # List specific options
    i_to_stub = {i: file for i, file in enumerate(incomplete)}
    for i in i_to_stub:
        p = i_to_stub[i]
        print(f'- Input {i} to remove stub {p} [{len(piece_info[p])}/{n}]')

    if len(incomplete) > 1:
        print('- To remove a combination of files, enter the numbers as space seperated (e.g., "0 1 3")')

    user_choice = cl_input('Your choice: ', valid=None)

    # Check if base option
    if user_choice == '':
        return None

    if user_choice == 'delete':
        return incomplete
    
    # Check if int
    try:
        return [i_to_stub[int(user_choice)]]

    # If not int, maybe combo of ints
    except ValueError:
        multiple = user_choice.split(' ')

        # Make sure all options are ints
        if len(multiple) > 1:

            try:
                indices = list(set([int(m) for m in multiple]))
                return [i_to_stub[i] for i in indices]

            except ValueError:
                print('Invalid choice!')
            
            except IndexError:
                print('One or more selected index are out of range!')

    except IndexError:
        print('Invalid choice seleted index is out of range!')

    # If here, then invalid choice, re-prompt.
    return _incomplete_prompt(incomplete, piece_info, n)

def incomplete_prompt(piece_info, n):

    # Get incomplete as list
    incomplete = get_incomplete(piece_info, n)

    if len(incomplete) == 0:
        return None

    print(f'Note: The following file stubs are not present across all subjects: {incomplete}')
    print()

    # Return the file stubs to remove
    return _incomplete_prompt(incomplete, piece_info, n)

def check_missing(subj_drs, subject_names, piece_info):
    '''Based on piece info, figure out if any subjects have no
    valid pieces, and remove them from the subj dr and subject names
    lists.'''

    removed = 0
    for dr, subj in zip(subj_drs.copy(), subject_names.copy()):

        # Check if in any pieces
        is_found = False
        for piece in piece_info:
            if subj in piece_info[piece]:
                is_found = True
                break

        # If not found, delete from dr's and names
        if not is_found:
            subj_drs.remove(dr)
            subject_names.remove(subj)
            removed += 1

    if removed > 0:
        print(f'Removed {removed} canidates(s) for no valid file stubs.')

    # Return updated directories and names
    return subj_drs, subject_names

def file_check(subj_drs, subject_names, check='dwi'):

    # Determine all unique pieces
    all_pieces, pieces_per_subj = set(), {}
    for s_dr, name in zip(subj_drs, subject_names):

        pieces = get_u_pieces(s_dr, name, check=check)

        # Keep track of union of all unique
        all_pieces = pieces.union(all_pieces)

        # Keep track per subjects
        pieces_per_subj[name] = pieces

    # Get info on each piece
    piece_info = {piece: set() for piece in all_pieces}
    for subj in pieces_per_subj:
        for piece in pieces_per_subj[subj]:
            piece_info[piece].add(subj)

    # Remove any subjects w/ no valid
    subj_drs, subject_names = check_missing(subj_drs, subject_names, piece_info)

    # Run functions designed to ask user about potentially identical
    # file stubs, just differ maybe from naming issue / typos
    n = len(subject_names)
    piece_info, known, done = check_combos(piece_info, n, known=None)
    while not done:
        piece_info, known, done = check_combos(piece_info, n, known=known)

    # Prompt user about any remaining incomplete
    to_remove = incomplete_prompt(piece_info, n)
    if to_remove is not None:

        # Remove any requested to be removed file stubs
        for stub in to_remove:
            del piece_info[stub]

        # Next, update the subj_drs / subject_names to remove
        # any with now no file stubs
        subj_drs, subject_names = check_missing(subj_drs, subject_names, piece_info)

    return piece_info, subj_drs, subject_names

def get_param_info(all_pieces):

    # Get param info from user for each piece
    param_info = {}
    for piece in all_pieces:
        print(f'For file stub == {piece}:')

        pe_dir = cl_input('What is the PE Direction (either + or -) : ', valid=['+', '-'])

        # Get readout time from user
        # TODO try to auto get
        read_out = None
        while read_out is None:
            try:
                read_out = float(input('What is the readout time?: ').rstrip())
            except ValueError:
                print('Invalid input!')
                read_out = None

        # Save to param info
        param_info[piece] = (pe_dir, read_out)

    return param_info

def prune_to_single_t1_stub(subj_drs, subject_names, anat_piece_info):

    if len(anat_piece_info) == 1:
        return subj_drs, subject_names, anat_piece_info

    print()
    print(f'Found {len(anat_piece_info)} different potential T1 file stubs, ' \
           'but we need to specify only a single T1 scan.')
    print('Please specify from the choices below which file stub ' \
          'represents the T1 scan to be used:')

    i_to_piece = dict(enumerate(anat_piece_info))
    for i in i_to_piece:
        print(f'- Input {i} to select {i_to_piece[i]}')

    # Prompt until get valid t1 stub
    t1_stub = None
    while t1_stub is None:
        try:
            t1_stub = i_to_piece[cl_input('Your choice: ', valid=int)]
        except IndexError:
            print(f'Index out of range! Please try again (between 0 and {len(i_to_piece)-1})')
        except KeyError:
            print('Invalid choice!')

    # Remove all but specified T1 stub
    for piece in list(anat_piece_info):
        if piece != t1_stub:
            del anat_piece_info[piece]

    # Update the subjects and dr's accordingly
    subj_drs, subject_names = check_missing(subj_drs, subject_names, anat_piece_info)

    return subj_drs, subject_names, anat_piece_info

def _gen_combos(dr, subj, piece, exts):

    combos = []
    for ext in exts:
        if isinstance(piece, tuple):
            for p in piece:
                combos.append(os.path.join(dr, subj + p + ext))
        else:
            combos.append(os.path.join(dr, subj + piece + ext))

    return combos

def _get_exists(dr, subj, piece, exts):

    combos = _gen_combos(dr, subj, piece, exts)
    for c in combos:
        if os.path.exists(c):
            return c

    raise RuntimeError(f'Error with {subj} determining real file location.')

def _get_subj_specific_piece(dr, subj, piece):
    '''Get the subject specific piece name, in case where piece may represent
    multiple names.'''

    # Easy case
    if not isinstance(piece, tuple):
        return piece

    # Otherwise try all pieces
    for p in piece:
        
        # Only need to check one file ext
        if os.path.exists(os.path.join(dr, subj + p + '.nii.gz')):
            return p

    # Maybe raise error instead
    raise RuntimeError(f'Error with {subj} determining relevant piece.')

def _proc_t1s(subjects, dwi_drs):

    # Note for including t1's for now can be okay that there are some missing
    anat_subjects = subjects.copy()

    # Get each of the anat folders
    # TODO handle case where dwi to anat isn't this clear-cut
    anat_drs = [f.replace('/dwi', '/anat') for f in dwi_drs]

    # Remove any with no files in anat, or no anat folder
    for s_dr, name in zip(anat_drs.copy(), anat_subjects.copy()):

        if not os.path.exists(s_dr) or len(os.listdir(s_dr)) == 0:
            anat_drs.remove(s_dr)
            anat_subjects.remove(name)

    # TODO Can / should single DTI no T1 subjects be run?
    n_missing = len(subjects) - len(anat_subjects)
    if n_missing > 0:
        print(f'Warning: {n_missing} canidate(s) w/ missing anat folder / files. ' \
               'These subjects will still be run, but without a T1 scan.')

    # Find any unique T1 stubs
    anat_piece_info, anat_drs, anat_subjects =\
        file_check(anat_drs, anat_subjects, check='t1')

    # Reduce to single t1 file stub
    anat_drs, anat_subjects, anat_piece_info =\
        prune_to_single_t1_stub(anat_drs, anat_subjects, anat_piece_info)

    assert len(anat_piece_info) == 1, "Something went wrong pruning to single t1 stub!"
    t1_key = list(anat_piece_info)[0]

    # Ask if skull_stripped, and get as bool
    skull_stripped_input = cl_input(f'Are T1 file stub {t1_key} ' \
                                    'already skull stripped? (y/n): ', ['y', 'n'])
    skull_stripped = skull_stripped_input == 'y'

    # Generate mapping from subject to (subject path, is skull_stripped)
    t1_mapping = {}
    gzip_warned = False
    for subj_dr, subj_name in zip(anat_drs, anat_subjects):
        existing_loc = _get_exists(subj_dr, subj_name, t1_key, VALID_T1_EXTS)

        # If existing location is not gziped, then we need to gzip it first
        if existing_loc.endswith('.nii'):

            # Let the user know why taking a long time
            if not gzip_warned:
                print('Compressing all t1.nii files to t1.nii.gz in place, please wait.')
                gzip_warned = True

            os.system(f'gzip {existing_loc}')
            existing_loc += '.gz'

        t1_mapping[subj_name] = (existing_loc, skull_stripped)

    return t1_mapping 

def proc_t1s(param_info, subjects, dwi_drs):

    # Ask about if to include T1s
    q = 'Should T1s be included? (y/n) - enter for default: '
    include_t1 = cl_input(q, valid=['y', 'n', ''])

    # Process default case, if two encodings, say no, if just +'s, say yes
    if include_t1 == '':
        u_pe_dirs = set([param_info[p][0] for p in param_info])

        include_t1 = 'n'
        if len(u_pe_dirs) == 1:
            include_t1 = 'y'

        print(f'Using default include_t1 = {include_t1}')
    print()

    # Include t1 case
    if include_t1 == 'y':
        return _proc_t1s(subjects, dwi_drs)

    # No case is just empty dict
    return {}


def _get_subj_drs(subj_name, pq_dr):
    '''Get the subj deriv directories w/ support for multi-session.'''

    # Multi-session case, we need seperate name
    # specifying that output should be two folders
    if '_' in subj_name:
        subj_folder = subj_name.replace('_', '/')
    else:
        subj_folder = subj_name

    # Make subject specific deriv folders, and ensure they exist
    subj_deriv_dr = os.path.join(pq_dr, subj_folder)
    subj_slurm_dr = os.path.join(subj_deriv_dr, 'slurm')
    subj_tmp_dr = os.path.join(subj_deriv_dr, 'tmp')
    os.makedirs(subj_slurm_dr , exist_ok=True)
    os.makedirs(subj_tmp_dr, exist_ok=True)

    # Return dict mapping
    return {'deriv': subj_deriv_dr,
            'slurm': subj_slurm_dr,
            'tmp': subj_tmp_dr}

def gen_config_files(dwi_drs, subjects, pq_dr,
                     rev_piece_info, param_info):

    config_files = {}
    for subj_dr, subj_name in zip(dwi_drs, subjects):

        # Get subject specific deriv directories
        s_drs = _get_subj_drs(subj_name, pq_dr)

        # Gen file, w/ write so overwrite if exists
        file_loc = os.path.join(s_drs['slurm'], 'dtiQA_config.csv')

        with open(file_loc, 'w', encoding="UTF-8") as file:

            # Proc each relevant piece for this subj
            for piece in rev_piece_info[subj_name]:

                # Get more specific piece and params and write to config file
                s_piece = _get_subj_specific_piece(subj_dr, subj_name, piece)
                pe_dir, read_out = param_info[piece]
                file.write(f'{subj_name + s_piece},{pe_dir},{read_out}\n')

        # And keep track in config_files dict
        config_files[subj_name] = file_loc

    return config_files

def gen_submit_files(dwi_drs, subjects, pq_dr, t1_mapping,
                     config_files, pe_axis):

    # Generate the slurm submit / command files
    # Load config and template
    config = load_config()
    template_lines = load_submit_template()

    # Process each subject, generating their slurm submit file
    submit_locs = []
    for subj_dr, subj_name in zip(dwi_drs, subjects):

        # Get subject specific deriv directories
        s_drs = _get_subj_drs(subj_name, pq_dr)

        # Get copy
        submit_lines = template_lines.copy()

        # Add output and error output lines
        output_ln = f'#SBATCH --output={os.path.join(s_drs["slurm"], "slurm_output.txt")}\n'
        error_ln = f'#SBATCH --error={os.path.join(s_drs["slurm"], "slurm_errors.txt")}\n'
        submit_lines.insert(1, output_ln)
        submit_lines.insert(2, error_ln)

        # Generate the full command
        command = 'singularity run '

        # Initial inputs singulatiry binding
        command += f'-B {subj_dr}:/INPUTS '

        # If T1, get stripped param, and add file binding
        stripped = 'raw'
        if subj_name in t1_mapping:
            t1_loc, is_stripped = t1_mapping[subj_name]
            if is_stripped:
                stripped = 'stripped'

            command += f'-B {t1_loc}:/INPUTS/t1.nii.gz '

        # Add rest of bindings
        command += f'-B {s_drs["deriv"]}:/OUTPUTS '
        command += f'-B {config_files[subj_name]}:/INPUTS/dtiQA_config.csv '
        command += f'-B {s_drs["tmp"]}:/tmp '
        command += f'-B {config["fs_license_loc"]}:/APPS/freesurfer/license.txt '

        # Add prequal loc and pe axis
        command += f'{config["prequal_simg_loc"]} {pe_axis} '

        # This is just skipped if no T1
        command += f'--synb0 {stripped} '

        # Add any extra params here
        # TODO, change anything here to be changable via config
        command += '--correct_bias on '

        # Add command line
        submit_lines.append(command)

        # Write submit file
        submit_loc = os.path.join(s_drs["slurm"], 'submit.sh')
        with open(submit_loc, 'w', encoding='UTF-8') as file:
            for line in submit_lines:
                file.write(line)

        submit_locs.append(submit_loc)

    return submit_locs

def main():
    '''Main script logic'''

    # Process passed arguments
    parser = argparse.ArgumentParser(description='Process commands for prequal proc dataset.')
    parser.add_argument('dataset', type=str, help='The name / location of the BIDs style dataset to process.')
    args = parser.parse_args()

    # Get full dataset path
    dataset_name = args.dataset
    dataset_dr = os.path.join(os.getcwd(), dataset_name)

    print('Starting prequal cmd line setup')
    print('Note: If you make a mistake you may exit this program at any time with ctrl-c.')
    print('')

    # Get subjects from dataset
    dwi_drs, subjects = get_subjects(dataset_dr, dataset_name)

    # First determine all unique dwi pieces, potentially updating drs / subjects lists
    piece_info, dwi_drs, subjects = file_check(dwi_drs, subjects, check='dwi')

    # Generate reversed piece info, from subject to pieces
    rev_piece_info = {}
    for piece in piece_info:
        subjs = piece_info[piece]
        for subj in subjs:
            try:
                rev_piece_info[subj].append(piece)
            except KeyError:
                rev_piece_info[subj] = [piece]

    # Get info on pieces from user
    print()
    param_info = get_param_info(piece_info)
    print()

    # Ask about the global phase encoding axis
    pe_axis = cl_input('What is the phase encoding (pe) axis '\
                       'of all images? (i/j, default==i): ', ['i', 'j', ''])
    if pe_axis == '':
        pe_axis = 'i'

    # Prompt and process any T1's
    t1_mapping = proc_t1s(param_info, subjects, dwi_drs)

    # Init output derivatives folder in the dataset
    pq_dr = init_pq_dr(dataset_dr)

    # Generate individual DTI config files in subjects deriv folders
    config_files = gen_config_files(dwi_drs, subjects, pq_dr,
                                    rev_piece_info, param_info)

    # Generate / save the submit files
    submit_locs = gen_submit_files(dwi_drs, subjects, pq_dr, t1_mapping,
                                   config_files, pe_axis)

    print(f'Prepared {len(submit_locs)} submissions in {pq_dr}/*/slurm/submit.sh')
    submit = cl_input('Submit all to cluster now? (y/n, default==y): ', ['y', 'n', ''])
    if submit in ['y', '']:
        for loc in submit_locs:
            os.system(f'sbatch {loc}')


if __name__ == '__main__':
    main()

