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
from sklearn.feature_extraction import text
from sklearn import svm

# Stop word list
######


class StopwordHandler():
    def __init__(self):
        self.stop_word_file = "english_added.txt"
        self.stop_word_set = None
        self.user_stop_word_list = []
    
    def get_user_stop_word_list(self):
        return self.user_stop_word_list
    
    def get_stop_word_set(self):
        if self.stop_word_set == None:
            f = open(self.stop_word_file)
            additional_stop_words = [word.strip() for word in f.readlines()]
            self.user_stop_word_list = additional_stop_words
            f.close()
            self.stop_word_set = text.ENGLISH_STOP_WORDS.union(additional_stop_words)
        return self.stop_word_set

stopword_handler = StopwordHandler()



word2vec_model = None
person_vec = ["Maria", "Smith", "James", "John", "Robert", "Michael", "William", "David", "Mary", "Patricia",\
                  "Linda", "Barbara", "Elizabeth", "Jennifer"]
semantic_vector_length = 300
OUTPUT_DIR = "data_output"
NBRS_MODEL_NAME = os.path.join(OUTPUT_DIR, "nbrs_model")

def get_saved_space_if_exists_and_file_names(file_name_path):
    file_name = os.path.basename(file_name_path)
    lines_f = NBRS_MODEL_NAME + "_current_lines_" + file_name
    next_dict_f = NBRS_MODEL_NAME + "_next_dict_"  + file_name
    
    saved_model = None
    if os.path.isfile(NBRS_MODEL_NAME + file_name) and os.path.isfile(lines_f)\
    and os.path.isfile(next_dict_f):
        saved_model = get_saved_model(NBRS_MODEL_NAME + file_name, lines_f, next_dict_f)

    return saved_model, file_name, lines_f, next_dict_f

def construct_vectors(lines, file_name_path):

    saved_model, file_name, lines_f, next_dict_f =\
        get_saved_space_if_exists_and_file_names(file_name_path)
    
    if saved_model != None:
        return saved_model
    print("Starts training a new nearest neighbour model\n******\n")
    current_lines = []
    current_lines_vector = []
    next_dict = {}

    lines = lines
    for prev_line, line, next_line in zip(lines, lines[1:], lines[2:]):
        if read_movie_lines.to_include([prev_line, line, next_line]):
            
            sentences = sent_tokenize(line)
            last_sentence_in_line = sentences[-1]
            word_list = word_tokenize(last_sentence_in_line)
            current_lines.append(line)
            word2vec_vector = get_final_vector_for_sentence(line, prev_line)
            current_lines_vector.append(word2vec_vector)
            if line not in next_dict:
                    next_dict[line] = []
            next_dict[line].append(next_line)

    print("lines ", len(current_lines))
    print("vectors ", len(current_lines_vector))
    print("unique lines", len(next_dict.keys()))

        #for el in current_lines_vector:

    X = np.array(current_lines_vector)
        #for el in X:

    print("Start traning nearest neigbhour")
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(X)
    print("Finish traning nearest neigbhour")

    joblib.dump(nbrs, NBRS_MODEL_NAME + file_name, compress=4)
    joblib.dump(current_lines, lines_f, compress=4)
    joblib.dump(next_dict, next_dict_f, compress=4)
    
    #joblib.dump(joblib.dump, , )
    return get_saved_model(NBRS_MODEL_NAME + file_name, lines_f, next_dict_f)

def get_saved_model(model_name, lines_f, next_dict_f):
    print("Load saved model. This might take a while....")
    saved_nbrs = joblib.load(model_name)
    saved_current_lines = joblib.load(lines_f)
    saved_next_dict = joblib.load(next_dict_f)
    print("Loaded saved model.")
    
    #distances, indices = nbrs.kneighbors(X)
    return saved_nbrs, saved_current_lines, saved_next_dict

def remove_bad_neighbours(closest_neighbours, already_used, is_last_line):
    resulting_neighbour_list = []
    for i in range(len(closest_neighbours)-1, -1, -1):
        if len(resulting_neighbour_list) == 0 and i == 0:
            # There are no alternative neighbours to use, so have to take this one regardle
            resulting_neighbour_list.append(closest_neighbours[i])
        elif closest_neighbours[i] in already_used:
            pass # Don't use this neighbour, since it is alreday in the dialogue
        elif is_last_line and closest_neighbours[i][-1] == "?":
            pass # if possible, don't end with a line with question mark
        elif is_last_line and closest_neighbours[i][-1] == "-":
            pass # if possible, don't end with a line with -
        else:
            resulting_neighbour_list.append(closest_neighbours[i])
    return resulting_neighbour_list

def get_nearest(nbrs, current_lines, next_dict, text_uncleaned, previous_line_uncleaned, already_used, is_last_line):
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

    # First, check line-pairs that are closest to a vector consisting of the current line and the previous one
    vec = get_final_vector_for_sentence(last_sentence_in_line, prev_last_sentence)

    neighbours = nbrs.kneighbors(np.array([vec]), 5, return_distance=False)[0]
    neighbours_distance = nbrs.kneighbors(np.array([vec]), 5, return_distance=True)[0][0]

    closest_neighbours = []
    next_lines = []
    first_dist = 2
    for index, dist in zip(neighbours, neighbours_distance):
        if dist < first_dist:
            first_dist = dist # Will only be set once in the loop

        if dist - first_dist < 0.08 or dist < 0.3:
            closest_neighbours.append(current_lines[index])

            next_lines.extend(next_dict[current_lines[index]])

    # Second, among the possible next-lines, check if there are any close neighbours
    smallest_distance_so_far = 2
    closest_neighbours = list(set(closest_neighbours))
    
    next_lines = remove_bad_neighbours(list(set(next_lines)), already_used, is_last_line)
    #next_lines = list(set(next_lines))[:10] # limit the options to not spend too much time to search
    next_line_ret = None
    
    next_line_matrix_vectors = []
    for next_line in next_lines:
        next_line_cleaned = clean(next_line)
        last_sentence_in_next_line = sent_tokenize(next_line_cleaned)[-1]
        current_and_next = get_final_vector_for_sentence(last_sentence_in_next_line, last_sentence_in_line)
        next_line_matrix_vectors.append(current_and_next)

    next_lines_matrix = np.array(next_line_matrix_vectors)
    clf = svm.OneClassSVM()
    clf.fit(next_lines_matrix)
    predicted = clf.predict(next_lines_matrix)
    outlier_class = 1
    nr_of_ones = 0
    nr_of_minus_ones = 0
    for el in predicted:
        if el == -1:
            nr_of_minus_ones = nr_of_minus_ones + 1
        else:
            nr_of_ones = nr_of_ones + 1
    if nr_of_minus_ones < nr_of_ones:
        outlier_class = -1
    next_lines_filtered_for_outliers = []
    if nr_of_ones > 0 and nr_of_minus_ones > 0: # If there's an outlier category
        for next_line, res in zip(next_lines, predicted):
            if res == outlier_class:
                pass # Don't add outliers to the candidate list
            else:
                next_lines_filtered_for_outliers.append(next_line)
    else:
        next_lines_filtered_for_outliers = next_lines

    for next_line in next_lines_filtered_for_outliers:
        next_line_cleaned = clean(next_line)
        last_sentence_in_next_line = sent_tokenize(next_line_cleaned)[-1]
        #Here the arguments are line (=last_sentence_in_next_line, last_sentence_in_line = prev_line)
        current_and_next = get_final_vector_for_sentence(last_sentence_in_next_line, last_sentence_in_line)
        neighbours_next = nbrs.kneighbors(np.array([current_and_next]), 5, return_distance=False)[0]
        neighbours_distance_next = nbrs.kneighbors(np.array([current_and_next]), 5, return_distance=True)[0][0]
        
        for index, dist in zip(neighbours_next, neighbours_distance_next):
            if dist < 0.4: # If it's very close, save time, by not going throug the list
                next_line_ret = next_line
                smallest_distance_so_far = dist
                break
            else:
                if dist < smallest_distance_so_far:
                    smallest_distance_so_far = dist
                    next_line_ret = next_line



    return closest_neighbours, next_line_ret

def use_space(file_name):
    saved_model, model_file_name, lines_f, next_dict_f =\
        get_saved_space_if_exists_and_file_names(file_name)
    
    if saved_model != None:
        return saved_model
    
    # TODO: Not using the entire corpus
    f = open(file_name)
    lines = [el.strip() for el in f.readlines()]
    print("read ", len(lines), " lines")
    
    nbrs, current_lines, next_dict = construct_vectors(lines, file_name)

    return nbrs, current_lines, next_dict


def read_beginnings(audience_file, other_output = None):
    first_lines = []
    rest_lines = {}
    if other_output:
        file_name = os.path.join(other_output, audience_file)
        print("Use examples from ", file_name)
        f = open(file_name)

    else:
        f = open(os.path.join(OUTPUT_DIR, audience_file))
    lines = f.readlines()
    lines = lines + [""]
    f.close()
    rest_of_dialog = []
    for i in range(0, len(lines)):
        line = lines[i]
        if line.strip() != "":
            rest_of_dialog.append(line.strip())
        if line.strip() == "":
            if len(rest_of_dialog) > 0:
                rest_lines[rest_of_dialog[0]] = rest_of_dialog
                first_lines.append(rest_of_dialog[0])
                rest_of_dialog = []
    return first_lines, rest_lines

def make_dialogs(nrs, file_name_1, file_name_2, audience_file, max_dialog_length, other_output=None):
    nbrs_1, current_lines_1,  next_dict_1 = use_space(file_name_1)
    print("Space 1 ready")
    nbrs_2, current_lines_2,  next_dict_2 = use_space(file_name_2)
    print("Space 2 ready")
    first_lines, rest_lines = read_beginnings(audience_file, other_output)
    
    random.shuffle(first_lines)
    nbrs = nbrs_2
    current_lines = current_lines_2
    next_dict = next_dict_2


    for n in range(0, nrs):
        already_used = []
        print("****************")
        first_line = first_lines[n]
        #print("first_line: ", first_line)
        rest = rest_lines[first_line]
        print(rest)
        generated_dialog_lines = []
        previous_line_to_compare_with = rest[0]
        line_to_compare_with = rest[1]
        name = "A-san: "
        print(name, previous_line_to_compare_with)
        name = "B-san: "
        print(name, line_to_compare_with)
        already_used.append(previous_line_to_compare_with)
        already_used.append(line_to_compare_with)
        
        generated_dialog_lines.append(previous_line_to_compare_with)
        generated_dialog_lines.append(line_to_compare_with)
        dialog_length = max(len(rest)-2, max_dialog_length)
        is_last_line = False
        for i in range(0, dialog_length):
            if i == dialog_length - 1:
                is_last_line = True
            if name == "A-san: ":
                name = "B-san: "
                nbrs = nbrs_2
                current_lines = current_lines_2
                next_dict = next_dict_2
            else:
                name = "A-san: "
                nbrs = nbrs_1
                current_lines = current_lines_1
                next_dict = next_dict_1
            
            closest_neighbour, next_line = get_nearest(nbrs, current_lines, next_dict,\
                                                    line_to_compare_with, previous_line_to_compare_with,\
                                                       already_used, is_last_line)
            print(name, next_line, "(Closest: ", closest_neighbour, ")")
            generated_dialog_lines.append(next_line)
            already_used.append(next_line)
            previous_line_to_compare_with = line_to_compare_with
            line_to_compare_with = next_line
            #print("line_to_compare_with", line_to_compare_with)
        name = "A"
        for generated, human in zip(generated_dialog_lines, rest):
            print(name + ": " + generated + " & " + name + ": " + human + "\\\\")
            if name == "A":
                name = "B"
            else:
                name = "A"
        print("\\\\")
        print()

def get_space():
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format("/Users/maria/mariaskeppstedtdsv/post-doc/gavagai/googlespace/GoogleNews-vectors-negative300.bin", binary=True)
    return word2vec_model

# The sentence needs to have at least three tokens
def get_vector_for_sentence(sentence):
    sentence = clean(sentence)
    sentence = sentence.replace(".", " ")
    #
    final_vector = [0, 0, 0, 0, 0]
    tokens = word_tokenize(sentence)
    if len(tokens) > 0 and tokens[-1] in "?!.,:;":
        if tokens[-1] == "?":
            final_vector = [1, 1, 1, 1, 1] # to bits to show its ending with question mark
        tokens = tokens[:-1]
    question_word_vector = [0, 0, 0, 0, 0, 0, 0]
    for i, question_word in enumerate(["who", "where", "when", "why", "what", "which", "how"]):
        for t in tokens:
            if t.lower() == question_word:
                question_word_vector[i] = 1
    final_vector = final_vector + question_word_vector*5

    # Take the first three and the last three words of the sentence to represent it. Regardless if they occur double
    if len(tokens) == 0:
        tokens = ["wernwrelkjdsfio", "wernwrelkjdsfio", "wernwrelkjdsfio", "wernwrelkjdsfio"] # words that don't exist, symbolises empty sentences
    if len(tokens) == 1:
        tokens =  list((tokens[0], tokens[0], tokens[0], tokens[0]))
    if len(tokens) == 2:
        tokens =  list((tokens[0], tokens[0], tokens[1], tokens[1]))
        #if len(tokens) == 3:
        #tokens =  list((tokens[0], tokens[1], tokens[1], tokens[1], tokens[2]))
    for token_nr, token in enumerate(tokens[:3] + tokens[-3:]):
        raw_vec = get_vector_for_token(token_nr, token, tokens)
        list_raw_vec = list(raw_vec)
        final_vector.extend(list_raw_vec)
    return final_vector

def get_vector_for_token(token_nr, token, all_tokens_in_sentence):
    token = token.strip()
    if len(token) > 1 and token[1].isupper(): # probably all is caps then
        token = token.lower()
    raw_vec = None
    try:
        if token_nr != 0 and token in word2vec_model:
            raw_vec = word2vec_model[token]
        elif token.lower() in word2vec_model:
            raw_vec = word2vec_model[token.lower()]
        elif token_nr != 0 and token[0].isupper():
            # Assume that the unknown token it is a name of a person
            random.shuffle(person_vec)
            raw_vec = word2vec_model[person_vec[0]]
        elif token.isdigit():
                raw_vec = word2vec_model["ten"]
        else:
                raw_vec = try_match_wider(token, all_tokens_in_sentence, token_nr, word2vec_model)
                    
    except KeyError:
        print("No vector found for", token)
        raw_vec = [0] * semantic_vector_length

    return raw_vec


def has_other_than_zero(vec):
    for el in vec:
        if el != 0:
            return True
    return False

# Returns the summed vector for all the words in a sentene (or rather, the average).
# Does a stop word filtering first
def get_summed_vector_for_sentence(sentence, extra_division_factor, remove_stop_words):
    sentence = clean(sentence)
    sentence = sentence.replace(".", " ")
    tokens = word_tokenize(sentence)
    summed_vector = semantic_vector_length*[0]
    nr_of_added = 0
    for token_nr, token in enumerate(tokens):
        if token not in stopword_handler.get_stop_word_set() or not remove_stop_words:
            # "token.lower(), token.lower(), token.lower()" is just an ugly fix, to be able to use
            # get_vector_for_token here as well, since this one looks an neighbours if the current
            # one is not in the word2model
            token_vec = get_vector_for_token(token_nr, token.lower(), [token.lower(), token.lower(), token.lower()])
            if has_other_than_zero(token_vec):
                summed_vector = np.sum([summed_vector, token_vec], axis=0)
                nr_of_added = nr_of_added + 1

    # Compute average, and at the same time, divide the weight by the "extra_division_factor", which can down-weight the importance
    # of the summed vector
    if nr_of_added > 0:
        average_vector = np.true_divide(summed_vector, nr_of_added)
    else:
        average_vector = summed_vector

    average_vector_weighted = np.true_divide(average_vector, extra_division_factor)
    return list(average_vector_weighted)

def get_final_vector_for_sentence(line, prev_line):
    extra_division_factor_previous = 1.1
    extra_division_factor_current = 1.3
    current_sentence_vector = get_vector_for_sentence(line)
    current_sentence_vector_average = get_summed_vector_for_sentence(line, extra_division_factor_current, remove_stop_words = False)
    prev_sentence_vector = get_summed_vector_for_sentence(prev_line, extra_division_factor_previous, remove_stop_words = True)
    combined_vector = current_sentence_vector + current_sentence_vector_average + prev_sentence_vector
    length = len(combined_vector)
    norm_vector = list(preprocessing.normalize(np.reshape(combined_vector, newshape = (1, length)), norm='l2')[0])

    """
    if len(norm_vector) != 2401:
        print("Wrong size of vector")
        print(sentence)
        print(norm_vector)
        exit(1)
    """
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
            raw_vec = word2vec_model[all_tokens[replace_index]]
            return raw_vec
    except IndexError:
        pass
    #print("No vector found for ", token, " and no vector for neighbouring token")
    raw_vec = [0] * semantic_vector_length
    return raw_vec

def clean(line):
    return line.replace(".", " . ").replace("'d", " would").replace("'s", " is").replace("'re", " are")\
    .replace('"', " ").replace("]", "").replace("[", "")\
    .replace("'ve", " have").replace("'ll", " will").replace("'m", " am")\
    .replace("-", " ").replace("...", ".").replace("..", ".").replace(",", " ")\
    .replace(" a ", " ").replace("A ", " ").replace(" of ", " ").replace("Of", "of")\
    .replace("'", " ").replace("`", "").replace(":", "").replace(";", "").replace("*", "")\
    .replace("`", " ").replace(" and ", " ").replace("And ", " ").replace(" to ", " ").replace("To ", " ")\


if __name__ == '__main__':
    #get_first_edit_distance_match("aabbccddeeffgg", None)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
 
    #print(stopword_handler.get_stop_word_set())

    print("Loading word2vec space into the memory. This takes a while ...")
    word2vec_model = get_space()
    print("Loaded the word2vec space")
    #get_vector_for_sentence("Why do you drink tea ?")

    # final evalution
    make_dialogs(262, os.path.join(OUTPUT_DIR, "a-san.txt"), os.path.join(OUTPUT_DIR, "b-san.txt"), "audience.txt", 4)

    # development data
    #make_dialogs(30, os.path.join(OUTPUT_DIR, "a-san.txt"), os.path.join(OUTPUT_DIR, "b-san.txt"), "evaluation_data.txt", 11)
    #make_dialogs(8, os.path.join(OUTPUT_DIR, "a-san.txt"), os.path.join(OUTPUT_DIR, "b-san.txt"), "interesting_examples.txt", 11, "pre_defined_output")

