#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <cmath>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <limits>
#include "csvstream.hpp"

using namespace std;

set<string> unique_words(const string &str){
    istringstream source(str);
    set<string> words;
    string word;
    while(source >> word){
        words.insert(word);
    }
    return words;
}

class Classifier{
private:
    int num_posts;
    set<string> all_labels;
    set<string> vocabulary;
    map<string, int> posts_per_label;
    map<string, int> word_count_total;
    map<string, map<string, int>> word_count_per_label;

    double log_likelihood(const string &word, const string &label) const{
        int count_C_w = 0;
        auto label_it = word_count_per_label.find(label);
        if(label_it != word_count_per_label.end()){
            auto word_it = label_it->second.find(word);
            if(word_it != label_it->second.end()){
                count_C_w = word_it->second;
            }
        }

        int count_C = posts_per_label.at(label);

        int count_w = 0;
        auto total_it = word_count_total.find(word);
        if(total_it != word_count_total.end()){
            count_w = total_it->second;
        }

        if(count_C_w > 0){
            return log(static_cast<double>(count_C_w) /count_C);
        }

        if(count_w > 0){
            return log(static_cast<double>(count_w) /num_posts);
        }

        if(num_posts == 0){
            return -numeric_limits<double>::infinity();
        }
        return log(1.0 /num_posts);
    }

    double log_prior(const string &label) const{
        if(posts_per_label.find(label) == posts_per_label.end() || num_posts == 0){
            return -numeric_limits<double>::infinity();
        }
        return log(static_cast<double>(posts_per_label.at(label)) / num_posts);
    }

public:
    Classifier() : num_posts(0){}

    void train_on_post(const string &label, const set<string> &words){
        num_posts++;
        all_labels.insert(label);
        posts_per_label[label]++;
        for(const string &word : words){
            vocabulary.insert(word);
            word_count_per_label[label][word]++;
            word_count_total[word]++;
        }
    }

    void print_summary() const{
        cout << "trained on " << num_posts << " examples" <<endl;
        cout << "vocabulary size = " << vocabulary.size() <<endl;
        cout << endl;
        cout << "classes:" << endl;
        for(const string &label : all_labels){
            double log_prior_val = log_prior(label);
            cout << "  " << label << ", "
                 << posts_per_label.at(label) << " examples, "
                 << "log-prior = " << log_prior_val << endl;
        }
        cout << endl;
        cout << "classifier parameters:" << endl;
        for(const string &label : all_labels){
            auto label_it = word_count_per_label.find(label);
            if(label_it != word_count_per_label.end()){
                for(const auto &word_pair : label_it->second){
                    const string &word = word_pair.first;
                    int count = word_pair.second;
                    double log_like_val =log_likelihood(word, label);
                    cout << "  " << label << ":" << word
                         << ", count = " << count
                         << ", log-likelihood = " << log_like_val <<endl;
                }
            }
        }
        cout << endl;
    }

    pair<string, double> predict(const set<string> &words) const{
        string best_label;
        double best_score = -numeric_limits<double>::infinity();
        for(const string &label : all_labels){
            double current_score = log_prior(label);
            for(const string &word : words){
                current_score += log_likelihood(word, label);
            }
            if(current_score >best_score){
                best_score = current_score;
                best_label = label;
            }
        }
        if(best_label.empty() && !all_labels.empty()){
            best_label = *all_labels.begin();
        }
        return {best_label, best_score};
    }
    int get_num_posts() const{
        return num_posts;
    }
};

void run_train_only_mode(const Classifier &classifier,
                         const vector<pair<string, string>> &training_posts){
    cout << "training data:" << endl;
    for(const auto &post : training_posts){
        cout << "  label = " << post.first
                << ", content = " << post.second << endl;
    }
    classifier.print_summary();
}

void run_test_mode(const Classifier &classifier, const string &test_filename){
    cout << "trained on " << classifier.get_num_posts() << " examples" << endl;
    cout << endl;
    csvstream test_csv(test_filename);
    int correct_predictions =0;
    int total_test_posts =0;
    map<string, string> csv_test_row;
    cout << "test data:" <<endl;
    while(test_csv >> csv_test_row){
        total_test_posts++;
        string correct_label = csv_test_row["tag"];
        string content = csv_test_row["content"];
        set<string> words = unique_words(content);
        pair<string, double> prediction = classifier.predict(words);
        string predicted_label =prediction.first;
        double score = prediction.second;
        if(predicted_label ==correct_label){
            correct_predictions++;
        }
        cout << "  correct = " << correct_label
                << ", predicted = " << predicted_label
                << ", log-probability score = " << score << endl;
        cout << "  content = " << content << endl;
        cout << endl;
    }
    cout << "performance: " << correct_predictions << " / "
            << total_test_posts << " posts predicted correctly" <<endl;
}
int main(int argc, char *argv[]){
    cout << setprecision(3);

    if(argc < 2 || argc >3){
        cout << "Usage: classifier.exe TRAIN_FILE [TEST_FILE]" <<endl;
        return 1;
    }

    string train_filename = argv[1];
    string test_filename;
    bool test_mode = (argc ==3);
    if(test_mode){
        test_filename = argv[2];
    }

    Classifier classifier;
    csvstream *train_csv = nullptr;
    vector<pair<string, string>> training_posts;

    try{
        train_csv = new csvstream(train_filename);
        map<string, string> csv_row;

        while(*train_csv >> csv_row){
            string label = csv_row["tag"];
            string content = csv_row["content"];
            training_posts.push_back({label, content});
            set<string> words = unique_words(content);
            classifier.train_on_post(label, words);
        }
        delete train_csv;
        train_csv =nullptr;

        if(!test_mode){
            run_train_only_mode(classifier,training_posts);
        }
        else{
            run_test_mode(classifier,test_filename);
        }
    }
    catch(const csvstream_exception &e){
        string failed_file;
        if(train_csv != nullptr){
            failed_file = test_filename;
            delete train_csv;
        }
        else{
            failed_file = train_filename;
        }
        cout << "Error opening file: " << failed_file <<endl;
        return 1;
    }
    return 0;
}
