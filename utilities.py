from IPython.display import display
import pandas as pd
import numpy as np
import time

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture


'''
Contents:
    01 compute_null_pct
    02 one_hot_encoder
    03 label_encoder
    04 expand_list_to_columns
    05 get_offer_engagements
    06 split_columns_by_offer_type
    07 get_transactions_allotment
    08 add_rfm_scores
    09 plot_pca_component
    10 transform_data
    11 fit_predict_data
    12 plot_optimization_analysis
    13 plot_silhouette_analysis
    14 plot_cluster_analysis
    15 compute_nil_pct
    16 compute_percentage_change
'''


def compute_null_pct(df):
    '''
    Print count and percentage of null values for dataframe columns

    Args:
    df (pd.DataFrame): subject dataframe

    Return:
    None
    '''

    # Create auxiliary dataframe to hold results, and
    # Get count of null values for dataframe columns
    aux_df = pd.DataFrame(df.isnull().sum()).reset_index().rename(
                             columns={'index': 'column', 0: 'count'})

    # Get percentage of null values for dataframe columns
    aux_df['pct'] = np.around(aux_df['count'] * 100 / df.shape[0], 2)

    # Show output
    print(aux_df.to_string(index=False))


def one_hot_encoder(df, target_column, categories, prefix):
    '''
    Encode categorical features as a one-hot numeric array with the
    extra ability to handle list values

    Args:
    df (pd.DataFrame): subject dataframe
    target_column (str): subject categorical column to be one-hot encoded
    categories (list): list of categories to search for in an array
    prefix (str): prefix to apply to one-hot encoded column labels

    Return:
    df (pd.DataFrame): copy of subject dataframe with categorical values
        in target_column one-hot encoded and target_column dropped
    '''

    # Create new column for each category and get binary values for each row
    # by searching for that category in target_column
    for i in categories:
        df[f'{prefix}_{i}'] = df[target_column].apply(lambda x: 1 if i in x else 0)

    # Drop target_column
    df = df.drop(target_column,axis=1)

    return df


def label_encoder(df, target_column, start=0, step=1):
    '''
    Encode labels in target_column with numeric values between 'start'
    and n_labels-1 at a specified 'step'

    Args:
    df (pd.DataFrame): subject dataframe
    target_column (str): subject column with labels to be encoded
    start (int) [default=0]: starting numeric value
    step (int) [default=1]: increment numeric value

    Return:
    df (pd.DataFrame): copy of subject dataframe with labels in target_column
        encoded with numeric values
    labels_dict (dict): map for labels and their encoded numeric values
    '''

    # Instantiate dictionary object to store numeric values for each label
    labels_dict = dict()

    # Loop over labels and encode with numeric values
    for label in df[target_column]:
        if label not in labels_dict:
            labels_dict[label] = start
            start += step

    # Map label keys in `target_column` with their values in `labels_dict`
    df[target_column] = df[target_column].apply(lambda x: labels_dict[x])

    return df, labels_dict


def expand_list_to_columns(df, target_column, column_labels):
    '''
    Expand an array of lists identical in length into separate columns

    Args:
    df (pd.DataFrame): subject dataframe
    target_column (str): subject column with an array of lists to be expanded
    column_labels (dict): labels for new columns as dictionary values in same
        order of list elements where keys start from 0 until n_elements-1

    Return:
    df (pd.DataFrame): copy of subject dataframe with list elements in each row
        of target_column expanded into separate columns

    '''

    # Split list elements into columns and merge back to original dataframe
    df = df[target_column].apply(pd.Series).merge(df, left_index=True, right_index=True)

    # Rename columns as per `column_labels`
    df = df.rename(columns=column_labels)

    # Drop `target_column`
    df = df.drop(columns=target_column)

    return df


def get_offer_engagements(aux_df, viewed_df, completed_df, transactions_df):
    '''
    Used inside pd.apply() along columns axis to extract for each offer (row)
    viewing and completion times from relevant input dataframes based on programmed
    logic then for each offer (row) return output_value of list type in the form
    [offer_viewed_time, offer_completed_time] with NaN inplace of any invaild element

    Args:
    aux_df (pd.DataFrame): subject dataframe containing received offers records
    viewed_df, completed_df (pd.DataFrame): input dataframes containing only
        records of viewed and completed offers respectively originating
        from `offers_df` split based on event type
    transactions_df (pd.DataFrame): input dataframe containing only
        transactions originating from `transcript` split based on value type

    Return:
    output_value (list): [offer_viewed_time, offer_completed_time] with NaN
        inplace of any invaild element based on programmed logic
    '''

    # Identify record parameters
    customer_id = aux_df['customer_id']
    offer_id = aux_df['offer_id']
    offer_start_time = aux_df['offer_start_time']
    offer_end_time = aux_df['offer_end_time']


    # PART A -- Get offer_viewed_time

    # Get view time from `viewed_df`
    viewed_time_list = viewed_df.loc[
        # Lookup the identified record
        (viewed_df['customer_id'] == customer_id) &
        (viewed_df['offer_id'] == offer_id) &
        # Assess view time is in duration
        (viewed_df['time'] >= offer_start_time) &
        (viewed_df['time'] <= offer_end_time)
    ]['time'].tolist()

    # Store view time if available in list else return NaN
    offer_viewed_time = np.NaN if not viewed_time_list else viewed_time_list[0]


    # PART B -- Get offer_completed_time

    # Get `time` from `completed_df` if offer is "bogo" or "disc"
    if aux_df['type_informational'] == 0:
        completed_time_list = completed_df.loc[
            # Lookup the identified record
            (completed_df['customer_id'] == customer_id) &
            (completed_df['offer_id'] == offer_id) &
            # Assess completion time is after view and in duration
            (completed_df['time'] >= offer_viewed_time) &
            (completed_df['time'] <= offer_end_time )
        ]['time'].tolist()

    # Get `time` from `transactions_df` if offer is "info"
    if aux_df['type_informational'] == 1:

        # Get 'time' for all customer transactions
        customer_transactions = transactions_df.loc[
            (transactions_df.customer_id == customer_id)
        ]['time'].tolist()

        # Instantiate list
        completed_time_list = []

        for transaction_time in customer_transactions:
            # Assess transaction time is after view and in duration
            if ((transaction_time >= offer_viewed_time) &
                (transaction_time <= offer_end_time)):
                completed_time_list.append(transaction_time)

    # Store completion time if available in list else return NaN
    offer_completed_time = np.NaN if not completed_time_list else completed_time_list[0]


    # Create return object
    output_value = [offer_viewed_time, offer_completed_time]

    return output_value


def split_columns_by_offer_type(df, target_columns):
    '''
    Split any giving columns by offer type (bogo, disc, info) and return
    new dataframe of only passed target_columns with `customer_id` as index

    Args:
    df (pd.DataFrame): subject dataframe
    target_columns (list): subject columns labels to splitted by offer type

    Return:
    output_df (pd.DataFrame): new dataframe with only passed target_columns
        splitted by offer type and `customer_id` as index
    '''

    # Create auxiliary list with `Grouper` columns
    aux_columns_list = ['customer_id', 'offer_id']

    # Append input `target_columns` to `aux_columns_list`
    for column in target_columns:
        aux_columns_list.append(column)

    # Create auxiliary dataframe pivoted by
    # `customer_id` as Grouper index and `offer_id` as Grouper columns
    aux_df = pd.pivot_table(
        df[aux_columns_list],
        index='customer_id', columns='offer_id',
        fill_value=0, aggfunc='mean')

    # Flatten multilevel column labels with suffix identifying offer_id (oid)
    aux_df.columns = aux_df.columns.map('{0[0]}_oid_{0[1]}'.format)

    # Create lists and store offer_id based on type (obtained from portfolio)
    bogo_oids = [1,2,4,9]
    disc_oids = [5,6,7,10]
    info_oids = [3,8]

    # Create output dataframe
    output_df = pd.DataFrame()

    for column in target_columns:

        # Instantiate variables to hold computations
        bogo_value = 0
        disc_value = 0
        info_value = 0

        # Compute for each offer type its value
        for i in bogo_oids:
            bogo_value += aux_df[f'{column}_oid_{i}']

        for i in disc_oids:
            disc_value += aux_df[f'{column}_oid_{i}']

        for i in info_oids:
            info_value += aux_df[f'{column}_oid_{i}']

        # Store computed values in relevant columns
        output_df[f'bogo_{column}'] = bogo_value
        output_df[f'disc_{column}'] = disc_value
        output_df[f'info_{column}'] = info_value

    return output_df


def get_transactions_allotment(df, aux_df, transactions_df):
    '''
    Used inside pd.apply() along columns axis to extract for each customer (row)
    count and amount of overall, promo, and nonpromo transactions in addition to
    recency for promo and nonpromo transactions required for RFM score computation
    later then for each customer (row) return output_value of list type in the form
    [txn_overall, amt_overall, txn_promo, amt_promo, txn_nonpromo, amt_nonpromo,
    recency_promo, recency_nonpromo] with NaN inplace of any invaild element

    Args:
    df (pd.DataFrame): subject dataframe containing records grouped by customers
    aux_df (pd.DataFrame): input dataframe containing received offers records
    transactions_df (pd.DataFrame): input dataframe containing only
        transactions originating from `transcript` split based on value type

    Return:
    output_value (list): [txn_overall, amt_overall, txn_promo, amt_promo,
        txn_nonpromo, amt_nonpromo, recency_promo, recency_nonpromo] with NaN
        inplace of any invaild element based on programmed logic
    '''

    # Identify customer
    customer_id = df['customer_id']

    # Get all transactions 'time' and `amount` for identified customer
    # This cerates list of lists in the form [time, amount]
    customer_transactions = transactions_df.loc[
        (transactions_df.customer_id == customer_id)
    ][['time', 'amount']].values.tolist()

    # Get `offer_start_time` and `offer_end_time` for all offers received
    # This cerates list of lists in the form [offer_start_time, offer_end_time]
    customer_offers = aux_df.loc[
        (aux_df.customer_id == customer_id)
    ][['offer_start_time', 'offer_end_time']].values.tolist()

    # Instantiate list for transactions overall
    overall = []

    # Instantiate list for transactions in promotional periods
    promo = []

    # For each list element in `customer_transactions` list
    for transaction in customer_transactions:
        # Append list element to `overall`
        overall.append(transaction)
        # Get transaction time for logical operation
        txn_time = transaction[0]

        # For each list element in `customer_offers` list
        for offer in customer_offers:
            # Get offer start and end times for logical operation
            offer_start_time = offer[0]
            offer_end_time = offer[1]
            # Assess transaction time during promotional periods and
            # Append list element to `promo`
            if txn_time in range(offer_start_time, offer_end_time + 1):
                promo.append(transaction)


    # PART A -- Get overall, promo, nonpromo transactions

    # Convert `overall` and `promo` to set of tuples
    ''' This serves two purposes:
    (1) Creates a set of unique elements since one transaction can be in
        two or more overlapping offer durations
    (2) Allows creating `nonpromo` by just subtracting other two sets'''
    overall = set(tuple(txn) for txn in overall)
    promo = set(tuple(txn) for txn in promo)
    nonpromo = overall - promo

    # Convert back to list to easily access values
    overall = list(overall)
    promo = list(promo)
    nonpromo = list(nonpromo)

    # Output "overall" transactions:
    # Get count of transactions by getting length of `overall`
    txn_overall = len(overall)
    # Get amount of transactions by summing second elements in `overall`
    amt_overall = np.round(
        sum(overall[i][1] for i in range(len(overall))), 2)

    # Output "promo" transactions:
    # Get count of transactions by getting length of `promo`
    txn_promo = len(promo)
    # Get amount of transactions by summing second elements in `promo`
    amt_promo = np.round(
        sum(promo[i][1] for i in range(len(promo))), 2)

    # Output "nonpromo" transactions:
    # Get count of transactions by getting length of `nonpromo`
    txn_nonpromo = len(nonpromo)
    # Get amount of transactions by summing second elements in `nonpromo`
    amt_nonpromo = np.round(
        sum(nonpromo[i][1] for i in range(len(nonpromo))), 2)


    # PART B -- Get recency for RFM score

    # Get recency_promo: time of most recent promo transaction
    recency_promo = np.NaN if not promo else promo[-1][0]

    # Get recency_nonpromo: time of most recent nonpromo transaction
    recency_nonpromo = np.NaN if not nonpromo else nonpromo[-1][0]


    # Create return object
    output_value = [txn_overall, amt_overall,
                    txn_promo, amt_promo,
                    txn_nonpromo, amt_nonpromo,
                    recency_promo, recency_nonpromo]

    return output_value


def add_rfm_scores(df, bins=5):
    '''
    Calculate the RFM score for both promo and nonpromo for each customers
    based on recency, frequency, and monetary values assuming they are already
    present in the dataset here specified in `value_columns` list

    Args:
    df (pd.DataFrame): subject dataframe
    bins (int) [default=5]: number of bins to rank values in.

    Return:
    df (pd.DataFrame): copy of subject dataframe with two new columns
    rfm_promo_score and rfm_nonpromo_score while `value_columns` dropped
    '''

    # Create a list of already-existing columns to be ranked
    value_columns = [
        'recency_promo', 'frequency_promo', 'monetary_promo',
        'recency_nonpromo', 'frequency_nonpromo', 'monetary_nonpromo']

    # Create a list of new columns to hold the result of ranking
    rank_columns = [
        'recency_promo_rank', 'frequency_promo_rank', 'monetary_promo_rank',
        'recency_nonpromo_rank', 'frequency_nonpromo_rank', 'monetary_nonpromo_rank']

    # Rank values in `value_columns` and store their rank in `rank_columns`
    for rank, value in zip(rank_columns, value_columns):
        df[rank] = ((pd.qcut(
            df[value], bins, labels=False) + 1).fillna(0)).astype(int)

    # Calculate RFM score for promo
    df['rfm_promo_score'] = np.round(
        ((df[rank_columns[0]] +
        df[rank_columns[1]] +
        df[rank_columns[2]]) / 3), 2)

    # Calculate RFM score for nonpromo
    df['rfm_nonpromo_score'] = np.round(
        ((df[rank_columns[3]] +
        df[rank_columns[4]] +
        df[rank_columns[5]]) / 3) , 2)

    # Drop `value_columns` and `rank_columns`
    df = df.drop(columns=value_columns)
    df = df.drop(columns=rank_columns)

    return df


def plot_pca_component(df, pca, component, n_features=5):
    '''
    Plot top n_features head and tail weights of PCA() class
    with each weight mapped to their corresponding column label

    Args:
    df (pd.DataFrame): subject dataframe
    pca (class): PCA()
    component (int): index of `n_components` to investigate
    n_features (int) [default=5]: top head and tail weights to plot

    Return:
    None
    '''

    # Get sorted weights for specified `component` by fisrt creating
    # dataframe with all components and mapped column labels
    weights = pd.DataFrame(pca.components_, columns=list(df.columns)).iloc[
        component].sort_values(ascending=False)

    # Create auxiliary dataframe with head and tail n_features and plot them
    aux_df = pd.concat([weights.head(n_features), weights.tail(n_features)])
    plt.figure(figsize=(17,4))
    aux_df.plot(kind='barh', title=f'Principal Component {component}')
    ax = plt.gca()
    ax.set_xlabel('Weight')

    # Show output
    plt.show()


def transform_data(df, ev, tsne=False):
    '''
    Apply PowerTransformer(), PCA(), and optionally TSNE() sequentially on dataframe

    Args:
    df (pd.DataFrame): subject dataframe
    ev (int, float, None, str): explained variance correspond to `n_components`
        parameter in PCA() class and hence inherits its arguments
    tsne (bool) [default=False]: When True, apply TSNE() on dataframe

    Return:
    X (array): transformed dataframe
    '''

    X = PCA(ev, random_state=42).fit_transform(PowerTransformer().fit_transform(df))

    if tsne == True:
        perplexity = int(X.shape[0] ** 0.5)
        X = TSNE(perplexity=perplexity, random_state=42).fit_transform(X)

    return X


def fit_predict_data(X, n_clusters, est='KMeans'):
    '''
    Estimate model parameters and predict labels for input X array

    Args:
    X (array): input data
    n_clusters (int): number of clusters to form
    est (str) [default='KMeans']: estimator to use; 'KMeans' or 'GaussianMixture'

    Return:
    model (self): fitted estimator
    labels (array): cluster labels
    '''

    est_dict = {
        'KMeans': KMeans(n_clusters, random_state=42),
        'GaussianMixture': GaussianMixture(n_clusters, random_state=42)}

    model = est_dict[est]
    labels = model.fit_predict(X)

    return model, labels


def plot_optimization_analysis(df, ev, est='KMeans', tsne=False, sample_size=0.05):
    '''
    Plot chnage across number of clusters ranging between 2 and 30 clusters in
    average silhouette score and sum of squared errors when est is 'KMeans'
    or 4 trials in average silhouette score when est is `GaussianMixture`

    Args:
    df (pd.DataFrame): subject dataframe
    ev (int, float, None, str): explained variance correspond to `n_components`
        parameter in PCA() class and hence inherits its arguments
    est (str) [default='KMeans']: estimator to use; 'KMeans' or 'GaussianMixture'
    tsne (bool) [default=False]: When True, apply TSNE() on dataframe
    sample_size (float) [default=0.05] = size of randomly selected sample

    Return:
    None
    '''

    # Apply transformations to data
    X = transform_data(df, ev, tsne)

    # Create clusters range list
    n_clusters = list(range(2, 31))

    # Plot for 'KMeans' est
    if est == 'KMeans':
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17,7))

        # Select random sample
        sample = X[np.random.choice(
            X.shape[0], int(X.shape[0] * sample_size), replace=False)]

        # Instantiate lists to store scores across clusters range
        sil_scores = []
        sse_scores = []

        # For each `n_clusters`; fit model, predict labels, append scores
        for i in n_clusters:
            model, labels = fit_predict_data(sample, i, est)
            sil_scores.append(silhouette_score(sample, labels))
            sse_scores.append(np.abs(model.score(X)))

        # Plot change in silhouette average (silhouette method)
        ax1.plot(n_clusters, sil_scores, linestyle='-', marker='o')
        ax1.locator_params(axis='x', nbins=31)
        ax1.set_title('Silhouette Method')
        ax1.set_xlabel('Clusters')
        ax1.set_ylabel('Silhouette Score')

        # Plot chnage in SSE to apply elbow method
        ax2.plot(n_clusters, sse_scores, linestyle='-', marker='o')
        ax2.locator_params(axis='x', nbins=31)
        ax2.set_title('Elbow Method')
        ax2.set_xlabel('Clusters')
        ax2.set_ylabel('Sum of Squared Errors')

        # Set `transformers`; to be used in plot suptitle adjunct text
        transformers = 'PowerTransformer -> PCA'

    # Plot for 'GaussianMixture' est
    if est == 'GaussianMixture':
        fig, ax = plt.subplots(2, 2, figsize=(17,7))

        # Plot 4 trials
        for trial in list(range(1,5)):

            # Select random sample
            sample = X[np.random.choice(
                X.shape[0], int(X.shape[0] * sample_size), replace=False)]

            # Instantiate list to store scores across clusters range
            sil_scores = []

            # For each `n_clusters`; fit model, predict labels, append scores
            for i in n_clusters:
                model, labels = fit_predict_data(sample, i, est)
                sil_scores.append(silhouette_score(sample, labels))

            # Plot change in silhouette average (silhouette method)
            ax = plt.subplot(2, 2, trial)
            ax.plot(n_clusters, sil_scores, linestyle='-', marker='o')
            ax.locator_params(axis='x', nbins=31)
            ax.set_title(f'Trial {trial}')
            ax.set_xlabel('Clusters')
            ax.set_ylabel('Silhouette Coefficient')

            # Set `transformers`; to be used in plot suptitle adjunct text
            transformers = 'PowerTransformer -> PCA -> TSNE'

    fig.tight_layout()

    # Add suptitle and adjunct text
    fig.suptitle(f'Optimal Clusters Analysis for {est} Clustering', size=17)
    fig.subplots_adjust(top=0.86)
    fig.text(0.5, 0.92,
             f'Transformers: {transformers} | PCA EV: {ev} | Sample Size: {sample_size}',
             ha='center',
             size=14)

    # Show output
    plt.show()


def plot_silhouette_analysis(df, ev, n_clusters, est='KMeans', tsne=False):
    '''
    Plot silhouette plot and feature space plot on two-columns axes

    Code in this function is obtained from scikit-learn example code
    titled "plot_kmeans_silhouette_analysis" with some minor changes
    scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

    Args:
    df (pd.DataFrame): subject dataframe
    ev (int, float, None, str): explained variance correspond to `n_components`
        parameter in PCA() class and hence inherits its arguments
    n_clusters (int): number of clusters to form
    est (str) [default='KMeans']: estimator to use; 'KMeans' or 'GaussianMixture'
    tsne (bool) [default=False]: When True, apply TSNE() on dataframe

    Return:
    None
    '''

    # Apply transformations to data
    X = transform_data(df, ev, tsne)

    # fit model, predict labels, and append overall average
    # score and silhouette score for each sample
    model, labels = fit_predict_data(X, n_clusters, est)
    sil_score = silhouette_score(X, labels)
    sil_sample = silhouette_samples(X, labels)

    # Create subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17,7))


    # Part A -- silhouette plot

    # Set appropriate limits for x-axis and y-axis
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    y_lower = 10
    for i in range(n_clusters):
        # Get aggregated sorted silhouette scores for samples
        ith_cluster_silhouette_values = sil_sample[labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.Spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                           facecolor=color, edgecolor=color, alpha=0.7)

        # Label silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i), color = 'black', fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='circle'))

        # Compute new y_lower for next plot
        y_lower = y_upper + 10

    ax1.set_title('Cluster Silhouette Plot')
    ax1.set_xlabel('Silhouette Coefficient')
    ax1.set_ylabel('Cluster')

    # Plot vertical line for overall average silhouette score
    ax1.axvline(x=sil_score, color="red", linestyle="--")

    # Set appropriate ticks for x-axis and y-axis
    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])


    # Part B -- feature space plot

    # Plot 1st and 2nd feature space
    colors = cm.Spectral(labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, c=colors)

    # Set `transformers`; to be used in plot suptitle adjunct text
    transformers = 'PowerTransformer -> PCA -> TSNE'

    # Illuminate cluster centers if est is 'KMeans'
    if est == 'KMeans':
        centers = model.cluster_centers_
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", s=300, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        # Set `transformers`; to be used in plot suptitle adjunct text
        transformers = 'PowerTransformer -> PCA'

    ax2.set_title('Cluster Feature Space Plot')
    ax2.set_xlabel('1st Feature')
    ax2.set_ylabel('2nd Feature')

    fig.tight_layout()

    # Add suptitle and adjunct text
    plt.suptitle((f'Silhouette Analysis for {est} Clustering'), size=17)
    fig.subplots_adjust(top=0.86)
    fig.text(0.5, 0.92,
             f'Transformers: {transformers} - PCA: {ev} - n_clusters: {n_clusters}',
             ha='center',
             size=14)

    # Show output
    plt.show()


def plot_cluster_analysis(plot, df, target_columns, title, cluster_column='cluster', print_stats=True):
    '''
    Plot using seaborn library target_columns and print descriptive statistics
    grouped by cluster_column to analyse clusters formations

    How to use:
    (1) For plot='countplot'; target_columns is expected to be single str where
        in this case x=cluster_column and hue=target_columns
    (2) For plot='violinplot'; target_columns could be:
        (A) single str where in this case x=cluster_column and y=target_columns
        (B) list of two str(s) where in this case x=cluster_column and
            y=(target_columns[1]/target_columns[0])
    (3) For plot='scatterplot'; target_columns is expected to be list of two str(s)
        where x=target_columns[0] and y=target_columns[1]

    Args:
    plot (str): plot type; 'scatterplot', 'countplot', 'violinplot'
    df (pd.DataFrame): subject dataframe
    target_columns (str, list):
    title (str): plot title/suptitle
    cluster_column (str) [default='cluster'] = column holding cluster labels

    Return:
    None
    '''

    # Create copy of input dataframe to manipulate
    aux_df = df.copy()

    # Assess target_columns is a list
    if (isinstance(target_columns, list)):

        # Define x and y to be used in violinplot or scatterplot
        x, y = target_columns[0], target_columns[1]

        # Perform calculations used in violinplot
        aux_df[title] = (aux_df[y] / aux_df[x]).replace([np.nan, np.inf], 0)
        target_columns = title

    if plot == 'scatterplot':
        # Gte set of clusters labels
        n_clusters = np.unique(df[cluster_column]).tolist()

        # Create subplot with 1 row and len(n_clusters) columns
        fig, ax = plt.subplots(1, len(n_clusters), figsize=(17,5), sharey=True, sharex=True)

        for i in n_clusters:
            # Plot scatterplot for each cluster on its corresponding column axis
            sns.scatterplot(data=df.loc[df[cluster_column]==i], x=x, y=y, ax=ax[i], s=15)
            ax[i].set_title(f'Cluster {i}', size=11)

            # Remove redundant x-labels and y-labels, and
            # Replace them with one lable on each overall axis
            ax[i].set_xlabel('')
            ax[i].set_ylabel('')
            fig.text(0.5, 0.04, x, ha='center', size=11)
            fig.text(0.08, 0.5, y, va='center', rotation='vertical', size=11)

            # Add suptitle
            fig.suptitle(title, size=14)
            fig.subplots_adjust(top=0.86)

    if plot != 'scatterplot':
        plt.figure(figsize=(17,5))

        # Plot according to `plot` input
        if plot == 'countplot':
            sns.countplot(data=aux_df, x=cluster_column, hue=target_columns)
        if plot == 'violinplot':
            sns.violinplot(data=aux_df, x=cluster_column, y=target_columns)

        plt.title(title)
        plt.ylabel('')
        plt.xlabel('Cluster')

    # Prepare descriptive statistics table
    stats = aux_df.groupby(cluster_column)[target_columns].describe().reset_index()
    stats = stats[['cluster', 'mean', 'min', '50%', 'max']]
    stats.rename(columns={'50%': 'median'}, inplace=True)
    stats = np.round(stats, 2)

    # Show outputs
    plt.show()
    if print_stats == True:
        print(stats.to_string(index=False))


def compute_nil_pct(df, target_column, cluster_column='cluster'):
    '''
    Print percentages of nil values in an arry grouped by cluster_column primarily
    designed to assess customers who never made any transactions in a given period
    hence only tested on columns; 'txn_overall', 'txn_promo', and 'txn_nonpromo'

    Args:
    df (pd.DataFrame): subject dataframe
    target_column (str): column to investigate
    cluster_column (str) [default='cluster'] = column holding cluster labels

    Return:
    None
    '''

    # Gte set of clusters labels
    n_clusters = np.unique(df[cluster_column]).tolist()

    # Instantiate list to holde percentage values
    idle_pct = []

    # For each cluster
    for i in n_clusters:
        # Get normalized value_counts for target_column
        pct = df.loc[df[cluster_column]==i][target_column].value_counts(normalize=True)
        # Assess zero value exist
        if 0 in pct:
            # Append percentage of zero values to `idle_pct`
            idle_pct.append(pct[0])
        else:
            # Append 0 percent to `idle_pct`
            idle_pct.append(0)

    # Create output_df
    output_df = pd.DataFrame({'cluster': n_clusters, 'pct': idle_pct})
    output_df['pct'] = np.round(output_df.pct * 100, 1)

    # Show output
    print(output_df.to_string(index=False))


def compute_percentage_change(df, initial, final, cluster_column='cluster'):
    '''
    Print center measurements for initial and final arrays and thier percentage
    change grouped by cluster_column primarily designed to assess percentage
    chnage increase in average RFM scores from 'nonpromo' to 'promo' periods
    hence only tested on columns; 'rfm_nonpromo_score' and 'rfm_promo_score'

    Args:
    df (pd.DataFrame): subject dataframe
    initial, final (str): column labels with values to investigate
    cluster_column (str) [default='cluster'] = column holding cluster labels

    Return:
    None
    '''

    # Group by cluster_column and get center measurements
    initial_stats = df.groupby(cluster_column)[initial].describe()[
        ['mean', '50%']].rename(columns={'50%': 'median'})

    final_stats = df.groupby(cluster_column)[final].describe()[
        ['mean', '50%']].rename(columns={'50%': 'median'})

    # Create merged output_df
    output_df = pd.merge(initial_stats, final_stats,
                         right_index=True, left_index=True,
                         suffixes=('_initial', '_final'))

    # Round all data of output_df
    output_df = np.round(output_df, 2)

    # Get data arrays to make later steps cleaner
    initial_mean = output_df['mean_initial']
    final_mean = output_df['mean_final']
    initial_median = output_df['median_initial']
    final_median = output_df['median_final']

    # Perform calculations and store results in relevant columns
    output_df['mean_%chnage'] = np.round(
        (final_mean - initial_mean) / initial_mean * 100, 2)

    output_df['median_%chnage'] = np.round(
        (final_median - initial_median) / initial_median * 100, 2)

    # Reorder output_df
    output_df = output_df[
        ['mean_initial', 'mean_final', 'mean_%chnage',
        'median_initial', 'median_final', 'median_%chnage']]

    print(output_df.reset_index().to_string(index=False))
