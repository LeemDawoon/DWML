from classifier.SVDDClassifier import SVDDClassifier
from tunner.Tunner import Tunner

if __name__ == '__main__':
    train_data_path = './data/sample/train.csv'
    eval_data_path = './data/sample/eval.csv'
    test_data_path = './data/sample/test.csv'

    num_cols = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    cate_cols = ['workclass', 'education', 'marital_status', 'relationship']

    clf = SVDDClassifier()

    train_arr = clf.load_data(train_data_path, num_cols, cate_cols, label_col='income_bracket', train_label_val='<=50K')
    eval_arr, eval_label_arr = clf.load_data(eval_data_path, num_cols, cate_cols, label_col='income_bracket')
    test_arr, test_label_arr = clf.load_data(test_data_path, num_cols, cate_cols, label_col='income_bracket')

    print(train_arr.shape)
    print(eval_arr.shape)
    print(eval_label_arr.shape)

    params = {
        'svm_params': {
            'kernel': 'linear'
        }

    }
    print(clf.train(train_arr, params=params))
    print(clf.evaluate(eval_arr, eval_label_arr))
    tunner = Tunner(clf)
    tunner.tunning(eval_arr, eval_label_arr, 10, space)
