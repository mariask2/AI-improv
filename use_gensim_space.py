from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn import preprocessing
from nltk.tokenize import sent_tokenize
from nltk import word_tokenize
from sklearn.externals import joblib
import os.path
from random import randint
import gensim
import read_movie_lines
from scipy.spatial import distance
import random

word2vec_model = None
person_vec = ["Maria", "Smith", "James", "John", "Robert", "Michael", "William", "David", "Mary", "Patricia"\
                  "Linda", "Barbara", "Elizabeth", "Jennifer"]
semantic_vector_length = 300
OUTPUT_DIR = "data_output"
NBRS_MODEL_NAME = os.path.join(OUTPUT_DIR, "nbrs_model")

def get_saved_space_if_exists_and_file_names(file_name_path):
    file_name = os.path.basename(file_name_path)
    lines_f = NBRS_MODEL_NAME + "_current_lines_" + file_name
    vectors_f = NBRS_MODEL_NAME + "_current_vectors_" + file_name
    next_dict_f = NBRS_MODEL_NAME + "_next_dict_"  + file_name
    
    saved_model = None
    if os.path.isfile(NBRS_MODEL_NAME + file_name) and os.path.isfile(lines_f)\
    and os.path.isfile(vectors_f) and os.path.isfile(next_dict_f):
        saved_model = get_saved_model(NBRS_MODEL_NAME + file_name, lines_f, vectors_f, next_dict_f)
    return saved_model, file_name, lines_f, vectors_f, next_dict_f

def construct_vectors(lines, file_name_path):

    saved_model, file_name, lines_f, vectors_f, next_dict_f =\
        get_saved_space_if_exists_and_file_names(file_name_path)
    
    if saved_model != None:
        return saved_model
    print("Starts training a new nearest neighbour model\n******\n")
    current_lines = []
    current_lines_vector = []
    next_dict = {}

    for line, next_line in zip(lines, lines[1:]):
        if read_movie_lines.to_include([line, next_line]):
            sentences = sent_tokenize(line)
            last_sentence_in_line = sentences[-1]
            word_list = word_tokenize(last_sentence_in_line)
            if len(sent_tokenize(next_line)) == 1:
                current_lines.append(line)
                norm_vector = get_vector_for_sentence(line)
                current_lines_vector.append(np.array(norm_vector))
                if line not in next_dict:
                    next_dict[line] = []
                next_dict[line].append(next_line)

    print(len(current_lines))
    print(len(current_lines_vector))
    print(len(next_dict.keys()))

        #for el in current_lines_vector:
#print(el)
    X = np.array(current_lines_vector)
        #for el in X:
#print(el)
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(X)
    
    joblib.dump(nbrs, NBRS_MODEL_NAME + file_name, compress=9)
    joblib.dump(current_lines, lines_f, compress=9)
    joblib.dump(current_lines_vector, vectors_f, compress=9)
    joblib.dump(next_dict, next_dict_f, compress=9)
    
    #joblib.dump(joblib.dump, , )
    return get_saved_model(NBRS_MODEL_NAME + file_name, lines_f, vectors_f, next_dict_f)

def get_saved_model(model_name, lines_f, vectors_f, next_dict_f):
    print("Load saved model. This might take a while....")
    saved_nbrs = joblib.load(model_name)
    saved_current_lines = joblib.load(lines_f)
    saved_current_lines_vector = joblib.load(vectors_f)
    saved_next_dict = joblib.load(next_dict_f)
    print("Loaded saved model.")
    
    #distances, indices = nbrs.kneighbors(X)
    return saved_nbrs, saved_current_lines, saved_current_lines_vector, saved_next_dict

def get_nearest(nbrs, current_lines, current_lines_vector, next_dict, text_uncleaned, previous_line_uncleaned):
    text = clean(text_uncleaned)
    text_uncleaned = None
    last_sentence_in_line = sent_tokenize(text)[-1]
    tokens = word_tokenize(last_sentence_in_line)

    prev_text = clean(previous_line_uncleaned)
    previous_line_uncleaned = None
    prev_last_sentence_list = sent_tokenize(prev_text)
    prev_last_sentence = ""
    if len(prev_last_sentence_list) > 0:
        prev_last_sentence = prev_last_sentence_list[-1]

    nr_of_neighbours = 2
    if len(tokens) < 4 and prev_last_sentence != "":
        nr_of_neighbours = 2
    vec = get_vector_for_sentence(last_sentence_in_line)
    #print("\n************\n Nearest to: " + text + "\n--")
    neighbours = nbrs.kneighbors(np.array([vec]), 10, return_distance=False)[0]
    neighbours_distance = nbrs.kneighbors(np.array([vec]), 10, return_distance=True)[0][0]
    #print("neighbours", neighbours)
    #print("neighbours_distance", neighbours_distance)
    closest_neighbours = []
    next_lines = []
    for index, dist in zip(neighbours, neighbours_distance):
        if dist < 0.80 and (len(current_lines[index]) < 5 or dist > 0.0): # Don't use the exact same line, don't copy previos dialog:
            closest_neighbours.append(current_lines[index])
            next_lines.extend(next_dict[current_lines[index]])
    if len(closest_neighbours) == 0: #No neigbours close enough, ad the most close anyway
        closest_neighbours.append(current_lines[neighbours[0]])
        next_lines.extend(next_dict[current_lines[neighbours[0]]])

    closest_neighbours = list(set(closest_neighbours))
    next_lines = list(set(next_lines))[:10] # limit the options to not spend too much time to search
    next_line_ret = None
    for next_line in next_lines:
        next_line_cleaned = clean(next_line)
        last_sentence_in_next_line = sent_tokenize(next_line_cleaned)[-1]
        next_vec = get_vector_for_sentence(last_sentence_in_next_line)
        neighbours_next = nbrs.kneighbors(np.array([next_vec]), 10, return_distance=False)[0]
        neighbours_distance_next = nbrs.kneighbors(np.array([next_vec]), 10, return_distance=True)[0][0]
        for index, dist in zip(neighbours_next, neighbours_distance_next):
            if dist < 0.87 and (len(word_tokenize(next_line)) < 5 or dist > 0.0): # Don't use the exact same line, don't copy previos dialog
                next_line_ret = next_line

    if next_line_ret == None:
        next_lines_ret = next_lines[0]
    # When there are several options, take the one that in closest in style to the previous one
    # except when there is not previous sentence, then only take the first in the list of possible options
    """
    if prev_last_sentence == "":
        selected_next = randint(0, len(next_lines)-1)
        next_line = next_lines[selected_next]
        return closest_neighbours, next_line
    smallest_distance_so_far = 2
    next_line = next_lines[0][0]
    last_sentence_vector = get_vector_for_sentence(prev_last_sentence)
    for line in next_lines:
        if len(next_lines) > 1 and line == prev_last_sentence:
            continue # Using the exact same line gets boring
        candidate_vec = get_vector_for_sentence(line)
        dst = distance.euclidean(candidate_vec,last_sentence_vector)
        if dst <= smallest_distance_so_far:
            if dst == 2 or dst > 1.0:
            # Don't make it too similar
                smallest_distance_so_far = dst
                next_line = line
                #print(smallest_distance_so_far)
                #print("next_line", next_line)
                #print("prev_last_sentence", prev_last_sentence)
       """
    return closest_neighbours, next_line

def use_space(file_name):
    saved_model, model_file_name, lines_f, vectors_f, next_dict_f =\
        get_saved_space_if_exists_and_file_names(file_name)
    
    if saved_model != None:
        return saved_model
    
    f = open(file_name)
    lines = [el.strip() for el in f.readlines()]
    print("read ", len(lines), "line")
    
    nbrs, current_lines, current_lines_vector, next_dict = construct_vectors(lines, file_name)

    
    return nbrs, current_lines, current_lines_vector, next_dict


def read_beginnings():
    first_lines = []
    rest_lines = {}
    f = open(os.path.join(OUTPUT_DIR, "audience.txt"))
    lines = f.readlines()
    f.close()
    rest_of_dialog = []
    for prev_line, line in zip(lines, lines[1:]):
        if prev_line.strip() == "":
            if len(rest_of_dialog) > 0 and len(first_lines) > 0:
                rest_lines[first_lines[-1]] = rest_of_dialog
                rest_of_dialog = []
            first_lines.append(line.strip())
        if line.strip() != "":
            rest_of_dialog.append(line.strip())
    return first_lines, rest_lines

def make_dialogs(nrs, file_name_1, file_name_2):
    #nbrs_1, current_lines_1, current_lines_vector_1, next_dict_1 = use_space(file_name_1)
    nbrs_2, current_lines_2, current_lines_vector_2, next_dict_2 = use_space(file_name_2)
    #TODO. Temp solution
    nbrs_1, current_lines_1, current_lines_vector_1, next_dict_1 = nbrs_2, current_lines_2, current_lines_vector_2, next_dict_2
    first_lines, rest_lines = read_beginnings()
    
    nbrs = nbrs_1
    current_lines = current_lines_1
    current_lines_vector = current_lines_vector_1
    next_dict = next_dict_1
    
    for n in range(0, nrs):
        print("****************")
        selected_dialog = randint(0, len(first_lines)-1)
        first_line = first_lines[selected_dialog]
        print("first_line: ", first_line)
        rest = rest_lines[first_line]
        print(rest)
        name = "A-san: "
        print(name, first_line)
        previous_line_to_compare_with = ""
        line_to_compare_with = first_line
        for i in range(0, len(rest)):
            if name == "A-san: ":
                name = "B-san: "
                nbrs = nbrs_2
                current_lines = current_lines_2
                current_lines_vector = current_lines_vector_2
                next_dict = next_dict_2
            else:
                name = "A-san: "
                nbrs = nbrs_1
                current_lines = current_lines_1
                current_lines_vector = current_lines_vector_1
                next_dict = next_dict_1

            closest_neighbour, next_line = get_nearest(nbrs, current_lines, current_lines_vector, next_dict,\
                                                    line_to_compare_with, previous_line_to_compare_with)
            print(name, next_line, "(Closest: ", closest_neighbour, ")")
            previous_line_to_compare_with = line_to_compare_with
            line_to_compare_with = next_line
            #print("line_to_compare_with", line_to_compare_with)


def get_space():
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format("/Users/maria/mariaskeppstedtdsv/post-doc/gavagai/googlespace/GoogleNews-vectors-negative300.bin", binary=True)
    return word2vec_model

# The sentence needs to have at least three tokens
def get_vector_for_sentence(sentence):
    sentence = clean(sentence)
    sentence = sentence.replace(".", " ")
    #
    final_vector = [0]
    tokens = word_tokenize(sentence)
    if len(tokens) > 0 and tokens[-1] in "?!.,:;":
        if tokens[-1] == "?":
            final_vector = [1] # a bit to show its ending with question mark
        tokens = tokens[:-1]
    # Take the first three and the last three words of the sentence to represent it. Regardless if they occur double
    if len(tokens) == 0:
        tokens = ["wernwrelkjdsfio", "wernwrelkjdsfio", "wernwrelkjdsfio", "wernwrelkjdsfio"]
    if len(tokens) == 1:
        tokens =  list((tokens[0], tokens[0], tokens[0], tokens[0]))
    if len(tokens) == 2:
        tokens =  list((tokens[0], tokens[0], tokens[1], tokens[1]))
    if len(tokens) == 3:
        tokens =  list((tokens[0], tokens[1], tokens[1], tokens[1], tokens[2]))
    for token_nr, token in enumerate(tokens[:4] + tokens[-4:]):
        token = token.strip()
        raw_vec = None
        try:
            if token_nr != 0 and token in word2vec_model:
                raw_vec = word2vec_model[token]
            elif token.lower() in word2vec_model:
                raw_vec = word2vec_model[token.lower()]
            elif token_nr != 0 and token[0].isupper():
                # Assume that the unknown token it is a name of a person
                random.shuffle(person_vec)
                #print("Try to replace ", token, " with ",  person_vec[0])
                raw_vec = word2vec_model[person_vec[0]]
            elif token.isdigit():
                raw_vec = word2vec_model["ten"]
            else:
                raw_vec = try_match_wider(token, tokens, token_nr, word2vec_model)
    
        except KeyError:
            print("No vector found for", token)
            raw_vec = [0] * semantic_vector_length
        list_raw_vec = list(raw_vec)
        final_vector.extend(list_raw_vec)
    length = len(final_vector)
    norm_vector = list(preprocessing.normalize(np.reshape(final_vector, newshape = (1, length)), norm='l2')[0])
    if len(norm_vector) != 2401:
        print("Wrong size of vector")
        print(sentence)
        print(norm_vector)
        exit(1)
    return norm_vector

def try_match_wider(token, all_tokens, token_nr, word2vec_model):
    token_list = list(token)
    for i in range(0, len(token)-1):
        if token_list[i] == token_list[i + 1]:
            del token_list[i]
            distance_one = ''.join(token_list)
            if distance_one in word2vec_model:
                return word2vec_model[distance_one]
        token_list = list(token)

    token_list = list(token)
    frequent_english_letters = ['e', 't', 'a', 'o', 'i', 'n', 's', 'h', 'r', 'd', 'l', 'c', 'u', 'm', 'w', 'f', 'g', 'y', 'p', 'b']
    for i in range(0, len(token)):
        for l in frequent_english_letters:
            token_list[i] = l
            distance_one = ''.join(token_list)
            if distance_one in word2vec_model:
                return word2vec_model[distance_one]
            token_list = list(token)
    # assume it's a person regardless of position
    if token[0].isupper():
        random.shuffle(person_vec)
        raw_vec = word2vec_model[person_vec[0]]
        return raw_vec
    try:
        int(token)
        return word2vec_model["100"]
    except ValueError:
        pass

    # Take next or previous token instead
    try:
        if token_nr < len(all_tokens)/2:
            replace_index = token_nr + 1
        else:
            replace_index = token_nr - 1
        if all_tokens[replace_index] in word2vec_model:
            print("Replace ", token, " with ", all_tokens[replace_index])
            raw_vec = word2vec_model[all_tokens[replace_index]]
            return raw_vec
    except IndexError:
        pass
    print("No vector found for ", token, " and no vector for neighbouring token")
    raw_vec = [0] * semantic_vector_length
    return raw_vec

def clean(line):
    return line.replace(".", " . ").replace("'d", " would").replace("'s", " is").replace("'re", " are")\
    .replace('"', " ")\
    .replace("'ve", " have").replace("'ll", " will").replace("'m", " am")\
    .replace("-", " ").replace("...", ".").replace("..", ".").replace(",", " ")\
    .replace(" a ", " ").replace("A ", " ").replace(" of ", " ").replace("Of", "of")\
    .replace("'", " ").replace("`", "").replace(":", "").replace(";", "").replace("*", "")\
    .replace("`", " ").replace(" and ", " ").replace("And ", " ").replace(" to ", " ").replace("To ", " ")\


if __name__ == '__main__':
    #get_first_edit_distance_match("aabbccddeeffgg", None)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    print("Loading word2vec space into the memory. This takes a while ...")
    word2vec_model = get_space()
    print("Loaded the word2vec space")
    #get_vector_for_sentence("Why do you drink tea ?")

    make_dialogs(10, os.path.join(OUTPUT_DIR, "a-san.txt"), os.path.join(OUTPUT_DIR, "b-san.txt"))
