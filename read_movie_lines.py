import os
from nltk.tokenize import sent_tokenize
from nltk import word_tokenize
import use_gensim_space
OUTPUT_DIR = "data_output"

mind_dict = {}
result_dict = {"A-SAN" : [], "B-SAN": [], "AUDIENCE" : [], "EVALUATION_DATA" : []}
mappings = {0 : "A-SAN", 1 : "B-SAN"}
# Each movie should be assigned to one of the three minds
def which_mind(movie_id):
    if movie_id not in mind_dict:
        mind_dict[movie_id] = len(mind_dict.keys())%2
    return mind_dict[movie_id]

def read_converstion():
    line_dict = {}
    all_lines = open("cornell_movie-dialogs_corpus/movie_lines.txt", encoding="ascii", errors="replace")
    for line in all_lines:
        sp = line.strip().split("+++$+++")
        line_dict[sp[0].strip()] = sp[4]

    conv = open("cornell_movie-dialogs_corpus/movie_conversations.txt", encoding="ascii",  errors="replace")
    more_than_ten = 0
    evaluation_lines = 0
    for line in conv:
        sp = line.strip().split("+++$+++")
        lines = ([el.strip() for el in sp[3].replace("['","").replace("']", "").split("', '")])
        
        conversation_list = [line_dict[el].strip() for el in lines]
        # Include audience conversations of at least 6 lines to use a data
        # to compare with in a turing test
        if to_include(conversation_list) and len(lines) >= 6:
            result_dict["AUDIENCE"].append(conversation_list)
        elif to_include(conversation_list) and len(lines) >= 1 and evaluation_lines < 100:
            # Use a maximum of 100 lines as evaluation data
            result_dict["EVALUATION_DATA"].append(conversation_list)
            evaluation_lines = evaluation_lines + 1
        else: # not audience
            mind = which_mind(sp[2])
            more_than_ten = more_than_ten + 1
            result_dict[mappings[mind]].append(conversation_list)

    print("Total number of dialgos extracted ", more_than_ten)
    print("Belonging to " + str(len(mind_dict.keys())) + " films")
    for key, value in result_dict.items():
        print(key, len(value))

        f = open(os.path.join("data_output", key.lower() + ".txt"), "w")
        for dialog in value:
            for line in dialog:
                f.write(line + "\n")
            f.write("\n")
        f.close()

def to_include(lines):
    for line in lines:
        if ">" in line or "<" in line:
            return False
        line = use_gensim_space.clean(line)
        if line.strip() == "":
            return False
        sentences = sent_tokenize(line)
        if len(sentences) > 2:
            return False
        if len(sentences) > 1 and len(word_tokenize(sentences[0])) > 3:
            return False
        last_sentence = sentences[-1]
        last_sentence = last_sentence.replace(".", " ")
        words = word_tokenize(last_sentence)
        if len(words) > 13 or len(words) < 2:
            return False
        if len(words) < 3:
            # Remove sentences with very little content
            for word in words:
                if word != "?" and word != "!" and word != "I" and len(word) < 2\
                    and not word.lower().startswith("ye") and not word.lower().startswith("no"):
                    return False
    return True

if __name__ == '__main__':
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    read_converstion()
