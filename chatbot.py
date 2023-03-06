# PA7, CS124, Stanford
# v.1.0.4
#
# Original Python code by Ignacio Cases (@cases)
######################################################################
import util
import re  ### imported, as per Ed
import numpy as np
import string

# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    def __init__(self, creative=False):  
        # The chatbot's default name is `moviebot`.
        # TODO: Give your chatbot a new name.
        self.name = 'Ely/Janice/Shumann/Chris MovieBot'

        self.creative = creative
        self.opposite = {'dont', 'didnt', 'wont', 'wasnt', 'werent', 'hasnt', 'cant', 'shouldnt', 'never', 'not', 'hardly', 'seldom', 'rarely', 'cannot'}
        self.strong_words = {'love', 'hate', 'best', 'worst', 'amazing', 'wonderful', 'majestic', 'favorite', 'really', 'very', 'awful', 'terrible', 'cringe', 'awesome'}

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = util.load_ratings('data/ratings.txt')
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')

        ########################################################################
        # TODO: Binarize the movie ratings matrix.                             #
        ########################################################################

        # Binarize the movie ratings before storing the binarized matrix.
        self.ratings = self.binarize(ratings)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        ########################################################################
        # TODO: Write a short greeting message                                 #
        ########################################################################

        greeting_message = "Wassup"

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return greeting_message

    def goodbye(self):
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """
        ########################################################################
        # TODO: Write a short farewell message                                 #
        ########################################################################

        goodbye_message = "ai bet, cya"

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return goodbye_message

    ############################################################################
    # 2. Modules 2 and 3: extraction and transformation                        #
    ############################################################################

    def process(self, line):
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this class.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'

        :param line: a user-supplied line of text
        :returns: a string containing the chatbot's response to the user input
        """
        ########################################################################
        # TODO: Implement the extraction and transformation in this method,    #
        # possibly calling other functions. Although your code is not graded   #
        # directly based on how modular it is, we highly recommended writing   #
        # code in a modular fashion to make it easier to improve and debug.    #
        ########################################################################
        if self.creative:
            response = "I processed {} in creative mode!!".format(line)
            title_list = self.extract_titles(line)
            print(title_list)
            for title in title_list:
                mov_index = self.find_movies_by_title(title)
                print(mov_index)
        else:
            response = "I processed {} in starter mode!!".format(line)
            ## TEST 
            # print(self.sentiment)
            # print(self.extract_titles(line))
            # title_list = self.extract_titles(line)
            # for title in title_list:
            #     mov_index = self.find_movies_by_title(title)
            #     print(mov_index)
            # print(self.extract_sentiment(line))
            

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return response

    @staticmethod
    def preprocess(text):
        """Do any general-purpose pre-processing before extracting information
        from a line of text.

        Given an input line of text, this method should do any general
        pre-processing and return the pre-processed string. The outputs of this
        method will be used as inputs (instead of the original raw text) for the
        extract_titles, extract_sentiment, and extract_sentiment_for_movies
        methods.

        Note that this method is intentially made static, as you shouldn't need
        to use any attributes of Chatbot in this method.

        :param text: a user-supplied line of text
        :returns: the same text, pre-processed
        """
        ########################################################################
        # TODO: Preprocess the text into a desired format.                     #
        # NOTE: This method is completely OPTIONAL. If it is not helpful to    #
        # your implementation to do any generic preprocessing, feel free to    #
        # leave this method unmodified.                                        #
        ########################################################################

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return text

    def extract_titles(self, preprocessed_input):
        """Extract potential movie titles from a line of pre-processed text.

        Given an input text which has been pre-processed with preprocess(),
        this method should return a list of movie titles that are potentially
        in the text.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: list of movie titles that are potentially in the text
        """
        # May change in creative mode, but passes basic mode 
        pattern = '(\"[^\"]+\"|\'[^\']+\')'  
        pot_titles = re.findall(pattern, preprocessed_input)
        # remove the quotation marks 
        res = [title[1:-1] for title in pot_titles]
        return res

    def find_movies_by_title(self, title):
        """ Given a movie title, return a list of indices of matching movies.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list
        that contains the index of that matching movie.

        Example:
          ids = chatbot.find_movies_by_title('Titanic')
          print(ids) // prints [1359, 2716]

        :param title: a string containing a movie title
        :returns: a list of indices of matching movies
        """
        movie_ids = []            
        title = title.lower()
        edited_title = title  # for now
        has_year = False
        
        title_tokens = title.split(" ")
        # check if tokens have years 
        if re.match("\([0-9][0-9][0-9][0-9]\)", title_tokens[-1]):
            has_year = True

        if len(title_tokens) > 1 and (title_tokens[0] == "a" or 
                                      title_tokens[0] == "the" or 
                                      title_tokens[0] == "an"):
            if has_year:
                has_year = True 
                edited_title = " ".join(title_tokens[1:-1])  # merge, exclude the first and last token 
                edited_title = edited_title + ", " + title_tokens[0] + " " + title_tokens[-1]
            else:
                edited_title = " ".join(title_tokens[1:]) + ", " + title_tokens[0]
        
        # find if the title matches by iterating all possible match
        print(edited_title)
        for index in range(len(self.titles)):
            pot_match = self.titles[index][0].lower()
            # print(pot_match)
            if has_year:
                if edited_title == pot_match:
                    movie_ids.append(index)
            else:
                if edited_title == pot_match[:-7]:
                    movie_ids.append(index)   

            if self.creative:
                alternate_pattern = ".+\(a.k.a. .+"
                foreign_pattern = ".+\(.+\) ([0-9]{4})"

                if re.match(alternate_pattern, pot_match) or re.match(foreign_pattern, pot_match):
                    alternate_title = re.findall(".+\(a.k.a. (.+)\).+", pot_match)
                    foreign_title = re.findall(".+[\(a.k.a.+\)]? \((.+)\) \([0-9]{4}\)", pot_match)

                    if len(alternate_title) != 0:
                        alternate_title = alternate_title[0]
                    if len(foreign_title) != 0:
                        foreign_title = foreign_title[0]

                    print(alternate_title, pot_match)
                    print(foreign_title, pot_match)

                    if edited_title == alternate_title:
                        movie_ids.append(index)
                    elif edited_title == foreign_title:
                        movie_ids.append(index)

        
        return movie_ids

    def extract_sentiment(self, preprocessed_input):
        """Extract a sentiment rating from a line of pre-processed text.

        You should return -1 if the sentiment of the text is negative, 0 if the
        sentiment of the text is neutral (no sentiment detected), or +1 if the
        sentiment of the text is positive.

        As an optional creative extension, return -2 if the sentiment of the
        text is super negative and +2 if the sentiment of the text is super
        positive.

        Example:
          sentiment = chatbot.extract_sentiment(chatbot.preprocess(
                                                    'I liked "The Titanic"'))
          print(sentiment) // prints 1

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a numerical value for the sentiment of the text
        """
        # tokenize the input first
        punctuations = string.punctuation
        # # remove punctuations and split by space (list) and lowercase 
        # tokens = preprocessed_input.strip(punctuations).lower().split(" ")

        # # for now, sum the scores 
        # score, sentiment = 0, 0
        # for token in tokens:
        #     if token in self.sentiment:
        #         if self.sentiment[token] == "neg": 
        #             score -= 1
        #         else:
        #             score += 1
        
        # if score > 0:
        #     sentiment = 1
        # elif score == 0:
        #     sentiment = 0
        # else:
        #     sentiment = -1

        # return sentiment 
        sent_score = 0
        super_sauce = 0
        # get rid of titles (don't want title to affect sentiment)
        titles = self.extract_titles(preprocessed_input)
        text_updated = preprocessed_input
        for t in titles:
            text_updated = preprocessed_input.replace(t, '')
        # get rid of punctuation
        text_updated_two = re.sub('[%s]' % re.escape(punctuations), '', text_updated)
        # break text into individual words
        text_updated_three = text_updated_two.split()

        # initialize temporary variables in loop
        neg_next = 1
        score = 0
        # print(text_updated_three)
        for i, word in enumerate(text_updated_three):

            # stemming
            # stem_word = self.stemmer.stem(word, 0,len(word)-1)
            # negation word classified as negative sentiment (find all words ending in nt if not in our list)
            if word in self.opposite or re.search("nt$", word):  # or stem_word in self.opposite
                neg_next *= -1
                if i == len(text_updated_three) - 1:
                    sent_score *= neg_next
                continue
            if self.creative:
                if word in self.strong_words:
                    # add weight if strong "super" word is present (2 or -2 returned)
                    super_sauce = 1
            if word in self.sentiment:
                # positive sentiment detected
                if self.sentiment[word] == 'pos':
                    score = 1
                # negative sentiment detected
                elif self.sentiment[word] == 'neg':
                    score = -1
                # flip score for next word if negation present
                score *= neg_next
                # reset negation variable
                neg_next = 1
                # compute running sentiment score
                sent_score += score
                # break out of loop if successful (to avoid double counting if present tense verb also there)
                continue
            # get entire word besides last letter if it is 'd'
            if word[-1:] == 'd':
                present_tense_word = word[0:-1]
                if present_tense_word in self.sentiment:
                    # support for past tense verbs such as 'loved' and 'liked'
                    if self.sentiment[present_tense_word] == 'pos':
                        score = 1
                    # support for past tense verbs such as 'hated' and 'disliked'
                    if self.sentiment[present_tense_word] == 'neg':
                        score = -1
                    score *= neg_next
                    neg_next = 1
                    sent_score += score
                    continue
            if word[-2:] == 'ed':
                present_tense_ed_word = word[0:-2]
                if present_tense_ed_word in self.sentiment:
                    # support for past tense verbs such as 'enjoyed'
                    if self.sentiment[present_tense_ed_word] == 'pos':
                        score = 1
                    # futher support for past tense verbs ending in 'ed'
                    if self.sentiment[present_tense_ed_word] == 'neg':
                        score = -1
                    score *= neg_next
                    neg_next = 1
                    sent_score += score

        # super sauce is 0 unless super negative or positive sentiment is detected in creative mode
        if sent_score > 0:
            sent_score = 1 + super_sauce
        elif sent_score < 0:
            sent_score = -1 - super_sauce

        return sent_score

    def extract_sentiment_for_movies(self, preprocessed_input):
        """Creative Feature: Extracts the sentiments from a line of
        pre-processed text that may contain multiple movies. Note that the
        sentiments toward the movies may be different.

        You should use the same sentiment values as extract_sentiment, described

        above.
        Hint: feel free to call previously defined functions to implement this.

        Example:
          sentiments = chatbot.extract_sentiment_for_text(
                           chatbot.preprocess(
                           'I liked both "Titanic (1997)" and "Ex Machina".'))
          print(sentiments) // prints [("Titanic (1997)", 1), ("Ex Machina", 1)]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a list of tuples, where the first item in the tuple is a movie
        title, and the second is the sentiment in the text toward that movie
        """
        pass

    def find_movies_closest_to_title(self, title, max_distance=3):
        """Creative Feature: Given a potentially misspelled movie title,
        return a list of the movies in the dataset whose titles have the least
        edit distance from the provided title, and with edit distance at most
        max_distance.

        - If no movies have titles within max_distance of the provided title,
        return an empty list.
        - Otherwise, if there's a movie closer in edit distance to the given
        title than all other movies, return a 1-element list containing its
        index.
        - If there is a tie for closest movie, return a list with the indices
        of all movies tying for minimum edit distance to the given movie.

        Example:
          # should return [1656]
          chatbot.find_movies_closest_to_title("Sleeping Beaty")

        :param title: a potentially misspelled title
        :param max_distance: the maximum edit distance to search for
        :returns: a list of movie indices with titles closest to the given title
        and within edit distance max_distance
        """
        #### find title ####  
        title = title.lower()
        close_titles = []
        min_dist_so_far = max_distance + 1

        #### Iterate & Edit All Titles ####
        for index in range(len(self.titles)):
            (p_title, genre) = self.titles[index]
            # lowercased p_title 
            p_title = p_title.lower()
            # extract the title from p_title 
            year_pattern = ".+ \([0-9]{4}\)"
            # find if title has year 
            if re.match(year_pattern, p_title):
                # p_title = re.findall("(.+[^\s]),?\s?\([0-9]{4}\)", p_title)
                p_title = re.findall("(.+)\s.+", p_title)
                p_title = p_title[0]
                # print(p_title)
            
            #### find the edit distance ####

            # initialize matrix
            D = np.zeros((len(title)+1, len(p_title)+1), 'int')
            for j in range(len(p_title)+1):
                D[0][j] = j
            for i in range(len(title)+1):
                D[i][0] = i

            # recurrence relation 
            for i in range(1, len(title)+1):
                for j in range(1, len(p_title)+1):
                    subst_cost = 2
                    if title[i-1] == p_title[j-1]:
                        subst_cost = 0
                    # update matrix
                    D[i][j] = np.min([D[i-1][j]+1, D[i][j-1]+1, D[i-1][j-1]+subst_cost])
                    
            #### update close titles list ####
            if D[len(title)][len(p_title)] <= max_distance:
                if D[len(title)][len(p_title)] == min_dist_so_far:
                    close_titles.append(index)
                elif D[len(title)][len(p_title)] < min_dist_so_far:
                    min_dist_so_far = D[len(title)][len(p_title)]
                    close_titles = []
                    close_titles.append(index)

        return close_titles


        

    def disambiguate(self, clarification, candidates):
        """Creative Feature: Given a list of movies that the user could be
        talking about (represented as indices), and a string given by the user
        as clarification (eg. in response to your bot saying "Which movie did
        you mean: Titanic (1953) or Titanic (1997)?"), use the clarification to
        narrow down the list and return a smaller list of candidates (hopefully
        just 1!)

        - If the clarification uniquely identifies one of the movies, this
        should return a 1-element list with the index of that movie.
        - If it's unclear which movie the user means by the clarification, it
        should return a list with the indices it could be referring to (to
        continue the disambiguation dialogue).

        Example:
          chatbot.disambiguate("1997", [1359, 2716]) should return [1359]

        :param clarification: user input intended to disambiguate between the
        given movies
        :param candidates: a list of movie indices
        :returns: a list of indices corresponding to the movies identified by
        the clarification
        """
        response = []
        for candidate in candidates:
            title = self.titles[candidate][0]
            if clarification in title:
                response.append(candidate)
        return response






    ############################################################################
    # 3. Movie Recommendation helper functions                                 #
    ############################################################################

    @staticmethod
    def binarize(ratings, threshold=2.5):
        """Return a binarized version of the given matrix.

        To binarize a matrix, replace all entries above the threshold with 1.
        and replace all entries at or below the threshold with a -1.

        Entries whose values are 0 represent null values and should remain at 0.

        Note that this method is intentionally made static, as you shouldn't use
        any attributes of Chatbot like self.ratings in this method.

        :param ratings: a (num_movies x num_users) matrix of user ratings, from
         0.5 to 5.0
        :param threshold: Numerical rating above which ratings are considered
        positive

        :returns: a binarized version of the movie-rating matrix
        """
        ########################################################################
        # TODO: Binarize the supplied ratings matrix.                          #
        #                                                                      #
        # WARNING: Do not use self.ratings directly in this function.          #
        ########################################################################

        # The starter code returns a new matrix shaped like ratings but full of
        # zeros.
        # binarized_ratings = np.zeros_like(ratings)
        binarized_ratings = np.where(ratings > threshold, 1, -1)
        binarized_ratings = np.where(ratings == 0, 0, binarized_ratings)
        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return binarized_ratings

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array

        :returns: the cosine similarity between the two vectors
        """
        ########################################################################
        # TODO: Compute cosine similarity between the two vectors.             #
        ########################################################################
        norm_u, norm_v = 0, 0
        for n in u:
            norm_u += n ** 2
        for n in v:
            norm_v += n ** 2

        norm_u, norm_v = np.sqrt(norm_u), np.sqrt(norm_v)
        if norm_u == 0 or norm_v == 0:  
            return np.dot(u,v)
        similarity = np.dot(u, v) / (norm_u * norm_v)
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return similarity

    def recommend(self, user_ratings, ratings_matrix, k=10, creative=False):
        """Generate a list of indices of movies to recommend using collaborative
         filtering.

        You should return a collection of `k` indices of movies recommendations.

        As a precondition, user_ratings and ratings_matrix are both binarized.

        Remember to exclude movies the user has already rated!

        Please do not use self.ratings directly in this method.

        :param user_ratings: a binarized 1D numpy array of the user's movie
            ratings
        :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
          `ratings_matrix[i, j]` is the rating for movie i by user j
        :param k: the number of recommendations to generate
        :param creative: whether the chatbot is in creative mode

        :returns: a list of k movie indices corresponding to movies in
        ratings_matrix, in descending order of recommendation.
        """

        ########################################################################
        # TODO: Implement a recommendation function that takes a vector        #
        # user_ratings and matrix ratings_matrix and outputs a list of movies  #
        # recommended by the chatbot.                                          #
        #                                                                      #
        # WARNING: Do not use the self.ratings matrix directly in this         #
        # function.                                                            #
        #                                                                      #
        # For starter mode, you should use item-item collaborative filtering   #
        # with cosine similarity, no mean-centering, and no normalization of   #
        # scores.                                                              #
        ########################################################################

        # movie_index_not_watched = np.where(user_ratings == 0.0)
        # movie_index_watched = np.where(user_ratings != 0.0)
        movie_ratings_not_watched = []

        # for m_n_index in movie_index_not_watched[0]:
        for m_n_index in range(len(ratings_matrix)):
            if user_ratings[m_n_index] != 0:
                continue

            cur_movie_ratings = ratings_matrix[m_n_index]
            rating_prediction = 0

            for m_index in range(len(user_ratings)):
                if user_ratings[m_index] != 0:
                    comp_movie_ratings = ratings_matrix[m_index]
                    cos_sim = self.similarity(comp_movie_ratings, cur_movie_ratings)
                    rating_prediction += cos_sim * user_ratings[m_index]

            if rating_prediction != 0:
                rating_prediction = rating_prediction 
                movie_ratings_not_watched.append( (rating_prediction, m_n_index) )
        
        movie_ratings_not_watched = sorted(movie_ratings_not_watched, reverse=True)
        recommendations = [m_n_index for (rating, m_n_index) in movie_ratings_not_watched]
        
        # movie_ratings_not_watched = sorted(movie_ratings_not_watched, reverse=True)
        # recommendations = [m_n_index for (rating, m_n_index) in movie_ratings_not_watched if user_ratings[m_n_index] == 0]
        # print(movie_ratings_not_watched)

        # recommendations = []
        # movie_rating_set = []

        # for i, rating in enumerate(user_ratings):
        #     # user did not rate the movie already
        #     if rating == 0:
        #         cos_sim = np.zeros(len(user_ratings))
        #         for j, rating_by_all in enumerate(ratings_matrix):
        #             if user_ratings[j] != 0:
        #                 item_item_sim = self.similarity(ratings_matrix[i], rating_by_all)
        #                 cos_sim[j] = item_item_sim
        #         user_score = np.dot(cos_sim, user_ratings)
        #         movie_rating_set.append((i, user_score))

        # sorted_scores = sorted(movie_rating_set, key=lambda tup:tup[1], reverse=True)

        # for i in range(0, k):
        #     recommendations.append(sorted_scores[i][0])

        # ########################################################################
        # #                        END OF YOUR CODE                              #
        # ########################################################################
        # return recommendations
            
        # #############################################################################
        # #                             END OF YOUR CODE                              #
        # #############################################################################
        return recommendations[:k]

    ############################################################################
    # 4. Debug info                                                            #
    ############################################################################

    def debug(self, line):
        """
        Return debug information as a string for the line string from the REPL

        NOTE: Pass the debug information that you may think is important for
        your evaluators.
        """
        debug_info = 'debug info'
        return debug_info

    ############################################################################
    # 5. Write a description for your chatbot here!                            #
    ############################################################################
    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.
        """
        return """
        Your task is to implement the chatbot as detailed in the PA7
        instructions.
        Remember: in the starter mode, movie names will come in quotation marks
        and expressions of sentiment will be simple!
        TODO: Write here the description for your own chatbot!
        """


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')
