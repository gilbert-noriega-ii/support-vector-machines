import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC





def plot_svc_decision_boundary(svm_clf, xmin, xmax, X, y):
    '''
    This function plots the decision boundary line.
    '''
    #SVM Classifier model
    svm_clf = SVC(kernel="linear", C=float("inf"))
    svm_clf.fit(X, y)

    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]

    # At the decision boundary, w0*x0 + w1*x1 + b = 0
    # => x1 = -w0/w1 * x0 - b/w1
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]

    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    svs = svm_clf.support_vectors_
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')
    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)



def large_margin_classification(X,y):
    '''
    This function creates subplots to show the difference between
    a good and bad decision boundary line.
    '''
    
    # Bad models
    x0 = np.linspace(0, 5.5, 200)
    pred_1 = 5*x0 - 20
    pred_2 = x0 - 1.8
    pred_3 = 0.1 * x0 + 0.5

    #SVM Classifier model
    svm_clf = SVC(kernel="linear", C=float("inf"))
    svm_clf.fit(X, y)
    
    #creating subplots
    fig, axes = plt.subplots(ncols=2, figsize=(10,2.7), sharey=True)

    plt.sca(axes[0])
    plt.plot(x0, pred_1, "g--", linewidth=2)
    plt.plot(x0, pred_2, "m-", linewidth=2)
    plt.plot(x0, pred_3, "r-", linewidth=2)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris versicolor")
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Iris setosa")
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.legend(loc="upper left", fontsize=14)
    plt.axis([0, 5.5, 0, 2])

    plt.sca(axes[1])
    plot_svc_decision_boundary(svm_clf, 0, 5.5, X, y)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs")
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo")
    plt.xlabel("Petal length", fontsize=14)
    plt.axis([0, 5.5, 0, 2])
    plt.show()


def feature_scaling_sensitivity():
    '''
    This function shows the effects scaling has on SVM.
    '''
    Xs = np.array([[1, 50], [5, 20], [3, 80], [5, 60]]).astype(np.float64)
    ys = np.array([0, 0, 1, 1])
    svm_clf = SVC(kernel="linear", C=100)
    svm_clf.fit(Xs, ys)

    plt.figure(figsize=(9,2.7))
    plt.subplot(121)
    plt.plot(Xs[:, 0][ys==1], Xs[:, 1][ys==1], "bo")
    plt.plot(Xs[:, 0][ys==0], Xs[:, 1][ys==0], "ms")
    plot_svc_decision_boundary(svm_clf, 0, 6, Xs, ys)
    plt.xlabel("$x_0$", fontsize=20)
    plt.ylabel("$x_1$    ", fontsize=20, rotation=0)
    plt.title("Unscaled", fontsize=16)
    plt.axis([0, 6, 0, 90])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(Xs)
    svm_clf.fit(X_scaled, ys)

    plt.subplot(122)
    plt.plot(X_scaled[:, 0][ys==1], X_scaled[:, 1][ys==1], "bo")
    plt.plot(X_scaled[:, 0][ys==0], X_scaled[:, 1][ys==0], "ms")
    plot_svc_decision_boundary(svm_clf, -2, 2, Xs, ys)
    plt.xlabel("$x_0$", fontsize=20)
    plt.ylabel("$x'_1$  ", fontsize=20, rotation=0)
    plt.title("Scaled", fontsize=16)
    plt.axis([-2, 2, -2, 2])



def sensitivity_to_outliers(X,y):
    '''
    This functions shows how sensitive hard margins are to outliers.
    '''
    X_outliers = np.array([[3.4, 1.3], [3.2, 0.8]])
    y_outliers = np.array([0, 0])
    Xo1 = np.concatenate([X, X_outliers[:1]], axis=0)
    yo1 = np.concatenate([y, y_outliers[:1]], axis=0)
    Xo2 = np.concatenate([X, X_outliers[1:]], axis=0)
    yo2 = np.concatenate([y, y_outliers[1:]], axis=0)

    svm_clf = SVC(kernel="linear", C=10**9)
    svm_clf.fit(Xo2, yo2)

    fig, axes = plt.subplots(ncols=2, figsize=(10,2.7), sharey=True)

    plt.sca(axes[0])
    plt.plot(Xo1[:, 0][yo1==1], Xo1[:, 1][yo1==1], "bs")
    plt.plot(Xo1[:, 0][yo1==0], Xo1[:, 1][yo1==0], "yo")
    plt.text(0.3, 1.0, "Impossible!", fontsize=24, color="red")
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.annotate("Outlier",
                 xy=(X_outliers[0][0], X_outliers[0][1]),
                 xytext=(2.5, 1.7),
                 ha="center",
                 arrowprops=dict(facecolor='black', shrink=0.1),
                 fontsize=16,
                )
    plt.axis([0, 5.5, 0, 2])

    plt.sca(axes[1])
    plt.plot(Xo2[:, 0][yo2==1], Xo2[:, 1][yo2==1], "bs")
    plt.plot(Xo2[:, 0][yo2==0], Xo2[:, 1][yo2==0], "yo")
    plot_svc_decision_boundary(svm_clf, 0, 5.5, Xo2, yo2)
    plt.xlabel("Petal length", fontsize=14)
    plt.annotate("Outlier",
                 xy=(X_outliers[1][0], X_outliers[1][1]),
                 xytext=(3.2, 0.08),
                 ha="center",
                 arrowprops=dict(facecolor='black', shrink=0.1),
                 fontsize=16,
                )
    plt.axis([0, 5.5, 0, 2])
    plt.show()



# def large_vs_fewer_margin_violations(X, y):
#     '''
#     This functions shows the affects of having large and
#     small margin violations.
#     '''
#     svm_clf = Pipeline([
#             ("scaler", StandardScaler()),
#             ("linear_svc", LinearSVC(C=1, loss="hinge", random_state=42)),
#         ])

#     svm_clf.fit(X, y)

#     svm_clf.predict([[5.5, 1.7]])

#     scaler = StandardScaler()
#     svm_clf1 = LinearSVC(C=1, loss="hinge", random_state=42)
#     svm_clf2 = LinearSVC(C=100, loss="hinge", random_state=42)

#     scaled_svm_clf1 = Pipeline([
#             ("scaler", scaler),
#             ("linear_svc", svm_clf1),
#         ])
#     scaled_svm_clf2 = Pipeline([
#             ("scaler", scaler),
#             ("linear_svc", svm_clf2),
#         ])

#     scaled_svm_clf1.fit(X, y)
#     scaled_svm_clf2.fit(X, y)

#     # Convert to unscaled parameters
#     b1 = svm_clf1.decision_function([-scaler.mean_ / scaler.scale_])
#     b2 = svm_clf2.decision_function([-scaler.mean_ / scaler.scale_])
#     w1 = svm_clf1.coef_[0] / scaler.scale_
#     w2 = svm_clf2.coef_[0] / scaler.scale_
#     svm_clf1.intercept_ = np.array([b1])
#     svm_clf2.intercept_ = np.array([b2])
#     svm_clf1.coef_ = np.array([w1])
#     svm_clf2.coef_ = np.array([w2])

#     # Find support vectors (LinearSVC does not do this automatically)
#     t = y * 2 - 1
#     support_vectors_idx1 = (t * (X.dot(w1) + b1) < 1).ravel()
#     support_vectors_idx2 = (t * (X.dot(w2) + b2) < 1).ravel()
#     svm_clf1.support_vectors_ = X[support_vectors_idx1]
#     svm_clf2.support_vectors_ = X[support_vectors_idx2]

#     fig, axes = plt.subplots(ncols=2, figsize=(10,2.7), sharey=True)

#     plt.sca(axes[0])
#     plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^", label="Iris virginica")
#     plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs", label="Iris versicolor")
#     plot_svc_decision_boundary(svm_clf1, 4, 5.9, X, y)
#     plt.xlabel("Petal length", fontsize=14)
#     plt.ylabel("Petal width", fontsize=14)
#     plt.legend(loc="upper left", fontsize=14)
#     plt.title("$C = {}$".format(svm_clf1.C), fontsize=16)
#     plt.axis([4, 5.9, 0.8, 2.8])

#     plt.sca(axes[1])
#     plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
#     plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
#     plot_svc_decision_boundary(svm_clf2, 4, 5.99, X, y)
#     plt.xlabel("Petal length", fontsize=14)
#     plt.title("$C = {}$".format(svm_clf2.C), fontsize=16)
#     plt.axis([4, 5.9, 0.8, 2.8])
#     plt.show()