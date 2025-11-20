import hashlib
import pandas as pd


""" 

The goal of this module is to create a function which takes as input the dataframe, 
one or multiple columns to base the split on and a training fraction.

We want to solve the following issue:

- Randomness behaves differently on each OS and Python version.
Even with the same random_stat, different systems or library versions 
can produce slightly different splits. Which means that the model will train on different rows
and resultscannot be compared.

- Rows do not stay in the same split when new data arrives.
If we add/substract rows ten yesterday's train can become today's test row.
We will lose continuity and comparison breaks

- Feature transformations leak information across random splits.



SOLUTION:

We will make the split depend on the row itself, not on randomness.
We will take a row's unique identifier, turn it into a long numbner using a hash,
convert into a value between 0 and 1, compare that value to the train_fraction and 
assign either train or test accordingly.

More details on the hash function:
- It takes any input and turns it into a long sequence of numbers/letters 
in a way that is 100% deterministic always on all machines (same input => same output).
The raw output will be a chain of letters/numbers that is in hexadecimal (base 16)
and this can be converted to an integer using int(hash_hex, 16).
Then we use a fiwed modulus that will give a float between O and 1.


"""


def create_sample_column(df: pd.DataFrame,
                         key_cols,
                         train_fraction: float = 0.8,
                         new_col: str = "sample") -> pd.DataFrame:
    """
    Create a deterministic sample split column ("train"/"test") based on hashing.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    key_cols : str or list of str
        Column(s) used as primary keys for stable splitting.
    train_fraction : float (0 < x < 1)
        Fraction of rows to assign to "train".
    new_col : str
        Name of output column.

    Returns
    -------
    df : pd.DataFrame
        The same dataframe with a new deterministic sample column.
    """

    if isinstance(key_cols, str):
        key_cols = [key_cols]

    def hash_to_uniform(row):
        # merge the key columns into a single string
        key_string = "||".join(str(row[col]) for col in key_cols)

        # create SHA256 hash → hex → int
        hash_int = int(hashlib.sha256(key_string.encode("utf-8")).hexdigest(), 16)

        # scale to [0,1]
        return hash_int % 10**12 / 10**12

    # compute the hash-based score
    df[new_col] = df.apply(hash_to_uniform, axis=1)

    # assign the bucket
    df[new_col] = df[new_col].apply(
        lambda x: "train" if x < train_fraction else "test"
    )

    return df


df = create_sample_column(df, key_cols=["customer_id"], train_fraction=0.8)
df["sample"].value_counts()

