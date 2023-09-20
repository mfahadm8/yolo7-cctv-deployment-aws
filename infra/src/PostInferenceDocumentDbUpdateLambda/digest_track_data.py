import pandas as pd
from matplotlib import pyplot as plt
import seaborn
import os
from numpy import polyfit, polyval, power

def ingest_data(df):
    """
    Harmonize column names, raise error if not found
    """

    if 'Frame' not in df.columns:
        if 'frame_num' in df.columns:
            df['Frame'] = df['frame_num']
        else:
            raise ValueError("Frame column not found")

    if 'center-y' not in df.columns:
        if 'top' in df.columns and 'height' in df.columns:
            df['center-y'] = df['top']+df['height']/2
        else:
            raise ValueError("Y value column not found")

    if 'center-x' not in df.columns:
        if 'left' in df.columns and 'width' in df.columns:
            df['center-x'] = df['width']+df['width']/2
        else:
            raise ValueError("X value column not found")
    
    return df

def get_moving_ids(df, threshold, plot=True):
        """
        Isolate tracks that have movement

        TODO potentially modify to use slope rather than absolute change.
        """
        
        agg = df.set_index('object_id')['center-y'].groupby(by='object_id').aggregate(func=['max', 'min'])
        agg['diff']=agg['max']-agg['min']
        #objids = agg[agg['diff'] > agg['diff'].max()/10].reset_index()['object_id']
        objids = agg[agg['diff'] > threshold].reset_index()['object_id']
        #print(objids)
        
        if plot:
            fig = seaborn.relplot(data=df[df['object_id'].isin(objids)], x='frame_num', y='center-y', hue='object_id', palette="Set2")
            fig.set_titles('Moving Tracks')
        #plt.plot('frame_num', 'center-y', data=df)

        return objids.to_list()

def join_object_ids(indices):
    """
    Joins the selected indices into a single series.

    indices - array of tuples, formatted ([index], [min_range, max_range])
    """
    dat = [[] for i in range(len(indices))]
    for i in range(len(indices)):
        dat[i] = df[(df['object_id'].isin(indices[i]['indices']))&
                    (df['frame_num'] > indices[i]['min'])&
                    (df['frame_num'] < indices[i]['max'])]
        dat[i]['object_id'] = i
    return dat


def plot_joined(data):
    fig, ax = plt.subplots()
    indices = data['object_id'].unique()
    for i in indices:
        ax.plot('Frame', 'center-y', data=data.loc[data['object_id']==i])
    fig.legend(range(len(indices)))
    plt.show()

def write_to_csv(data, filename, ask_confirm=True):
    if os.path.splitext(filename)[1] == '':
        filename += '.csv'

    if ask_confirm:
        confirm = input("Write to file? ")
    else:
        confirm='y'

    if confirm.lower() != 'y':
        return False
    
    data.to_csv(filename, index=False, mode='a')
    #print(f"Dataset {i} written to {filename}")

def is_adjacent(dat1, dat2, distance=25):
    """
    Tests for data continuity between set end and start
    """
    return abs(dat1['center-x'].max() - dat2['center-x']) < distance and \
        abs(dat1['center-y'].max() - dat2['center-y']) < distance

def predict_next(data, x_value, tail_length=10):
    """
    Predict next point in data series
    """
    
    x_data = data['center-x'].iloc[-tail_length:]
    y_data = data['center-x'].iloc[-tail_length:]

    fit, residuals = polyfit(x_data, y_data, deg=2, full=True)

    print(fit, residuals)
    return polyval(fit, x_value)

def slice_flat_start(data, id, window=40, dy_threshold=25, deviation_threshold=3, tolerance=5, verbose=False):
    """
    Sometimes the track starts flat due to mistracking, then starts
    following a ball. We must cut off those flat areas.

    window: the number of points used to curve fit and check linearity
    """
    if window < 3:
        raise ValueError("Must have a window > 3, preferably 20+")
    if data is None:
        print("[Errror] Empty data set.")
        return data
    # is the beginning straight?
    objset = data.loc[data['object_id']==id]
    if len(objset) < window:
        return data
    
    print(f"Checking ID {id} for flatness...")
    #print(f'ID {id}, length {len(subset)}')
    
    flat_until = -1
    lastfit = None
    values = range(0, len(objset)-window, window)
    for i in values:
        subset = objset.iloc[i:i+window,:]
        fit, res, _r, _sv, _rcond = polyfit(x=subset['Frame'], y=subset['center-y'], deg=2, full=True)
        #print(fit)
        #print(res)

        # If flat in window - mark value and continue further on.    
        if abs(fit[1]) <= dy_threshold:
            # flat area
            flat_until=i+window
            lastfit = fit
            #print(id, 'flat until', i+window)
            continue
        elif flat_until > 0:
            # Flat before, but no longer flat - calculate part to slice
            subset = objset.iloc[0:i+window,:].copy()
            fit, res, _r, _sv, _rcond = polyfit(x=subset['Frame'], y=subset['center-y'], deg=2, full=True)
            std = subset.iloc[0:flat_until, subset.columns.get_loc('center-y')].std()

            subset['fit'] = polyval(fit, subset.loc[:,'Frame'])
            subset['residual'] = subset.loc[:,'center-y'] - subset.loc[:,'fit']
            
            #print(subset['residual'])
            last_flat = subset.loc[subset['residual'] < std*deviation_threshold, 'Frame'].max() - tolerance
                    
            
            if verbose:
                print("Last frame where flat:", last_flat)
                
                fig, ax = plt.subplots()
                ax.scatter(subset['Frame'], subset['center-y'], c='green')
                ax.plot(subset['Frame'], subset['fit'], 'b-')
                ax.set_ylabel('Y value', color='g')
                secax = ax.twinx()
                secax.scatter(subset['Frame'], subset['residual'], c='red')
                secax.set_ylabel('Residuals', color='r')
                fig.suptitle("Flatness check for ID "+str(id))

            #### Return dataframe with sliced series
            c1 = data['object_id'] == id
            c2 = data['Frame'] < last_flat
            c_all = c1 & c2
            
            return data[~c_all]
        
        if flat_until == -1:
            print(id, 'not flat.')
            return data

    return data

def match_ends(data, left_id, right_id, dist_threshold=0.4, frame_threshold=25):
    """
    Checks for end proximity of two id sets
    """

    left_data = data[data['object_id']==left_id]
    right_data = data[data['object_id']==right_id]

    if len(left_data) == 0 or len(right_data) == 0:
        raise ValueError("Empty dataset presented.")

    # Is there overlap, and how much
    mixed_data = pd.concat([left_data, right_data])
    overlap = len(mixed_data.duplicated(subset='Frame'))
    print(overlap, "frames have multiple values.")

    left_last_frame = right_data.loc[right_data['Frame'] == right_data['Frame'].min()]
    right_first_frame = left_data.loc[left_data['Frame'] == left_data['Frame'].max()]

    #print(left_last_frame['Frame'].values)
    #print(right_first_frame['Frame'].values)
    
    # How separated are the series, normalized to range
    fdist = (right_first_frame['Frame'].values[0] - left_last_frame['Frame'].values[0])
    xdist = (right_first_frame['center-x'].values[0] - left_last_frame['center-x'].values[0]) / \
            (data['center-x'].max()-data['center-x'].min())
    ydist = (right_first_frame['center-y'].values[0] - left_last_frame['center-y'].values[0]) / \
            (data['center-y'].max()-data['center-y'].min())

    dist = (xdist**2 + ydist**2)**0.5
    print(f'IDs: {left_id} and {right_id} Frame,X,Y distance {fdist},{xdist},{ydist}')
    
    return dist <= dist_threshold and fdist <= frame_threshold

def join_contiguous(data, ids):
    """
    Join data sets that seem continuous from their proximity
    """

    data = data[data['object_id'].isin(ids)]

    for i, id in enumerate(ids[:-1]):
        if match_ends(data, id, ids[i+1]):
            data.loc[data['object_id']==id,'object_id']=ids[i+1]

    return data

def digest_track_data(data, args):
    """
    Run through full process.

    -Harmonizes column format
    -Removes static (non-moving) objects
    -slices off misdetected flat starts
    -joins adjacent tracks

    """
    
    data = ingest_data(df)
    
    objids = get_moving_ids(data, args.threshold, plot=True)
    
    for id in objids:
        data = slice_flat_start(data, id, dy_threshold=args.threshold)
    
    data = join_contiguous(data, objids)

    if args.verbose:
        plot_joined(data)
        print(data)
    
    return data

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        prog="Trackable Ball track filter and joiner",
        description="Removes non-moving tracks and joins adjacent ones"
    )
    parser.add_argument("filename", type=str,
                        help="input csv file file with separate object_ids and positions")
    parser.add_argument("--outfile", default='', type=str,
                        help="Output CSV file")
    parser.add_argument("--threshold",
                        type=int, default=25,
                        help="minimum movement in px for detection")
    parser.add_argument("--verbose", action='store_true')
    args, unknown = parser.parse_known_args()

    #folder = '../Media/Calhoun Trackable/Soccer Launcher Side View/outputs/fullpath/'
    #fname = "IMG_9192_ball.csv"
    #filename = folder + fname

    df = pd.read_csv(args.filename, index_col=None)

    df = digest_track_data(df, args)

    if args.outfile != '':
        write_to_csv(df, args.outfile, ask_confirm=False)


