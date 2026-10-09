"""Extraction of standardized market-data tables from Excel workbooks."""

from pathlib import Path

import numpy as np
import pandas as pd


IMPLIED_VOLATILITY_SHEET = "implied_volatility"
VARIANCE_SWAP_SHEET = "variance_swaps"
VOLATILITY_SWAP_SHEET = "volatility_swaps"

IMPLIED_VOLATILITY_COLUMNS = [
    "maturity",
    "log_moneyness",
    "implied_volatility",
]

VARIANCE_SWAP_COLUMNS = [
    "maturity",
    "variance_swap_quote",
]

VOLATILITY_SWAP_COLUMNS = [
    "maturity",
    "volatility_swap_quote",
]


def _extract_sheet(excel_file, sheet_name, required_columns):
    """
    Read one standardized worksheet after checking its structure.

    Parameters
    ----------
    excel_file : str or pathlib.Path
        Path to the Excel workbook.

    sheet_name : str
        Exact, case-sensitive worksheet name to read.

    required_columns : sequence of str
        Column names that must be present in the worksheet.

    Returns
    -------
    data : pandas.DataFrame
        DataFrame read from the worksheet. Values and rows are returned as
        read by pandas; this function does not clean, sort, or validate the
        data values.

    Raises
    ------
    FileNotFoundError
        If excel_file is not an existing file.

    ValueError
        If sheet_name is not present, if one or more required columns are
        missing, or if the worksheet header contains duplicate column names.

    Notes
    -----
    The header is inspected before the DataFrame is returned so that duplicate
    names are detected even when pandas would otherwise disambiguate them by
    appending a suffix such as .1.
    """
    # 1. Check that the Excel file exists.
    path = Path(excel_file)
    if not path.is_file():
        raise FileNotFoundError(f"Excel file not found: {path}")

    # 2. Open the workbook and check that the requested worksheet exists.
    workbook = pd.ExcelFile(path)
    try:
        if sheet_name not in workbook.sheet_names:
            raise ValueError(
                f"Worksheet {sheet_name!r} is not present in {path}."
            )

        # 3. Inspect the raw header before pandas can disambiguate duplicates.
        raw_header = pd.read_excel(
            workbook,
            sheet_name=sheet_name,
            header=None,
            nrows=1,
        )

        if raw_header.empty:
            header_labels = []
        else:
            header_labels = [
                label
                for label in raw_header.iloc[0].tolist()
                if not pd.isna(label)
            ]

        # 4. Reject duplicate column names.
        seen_labels = set()
        duplicate_labels = []
        for label in header_labels:
            if label in seen_labels and label not in duplicate_labels:
                duplicate_labels.append(label)
            seen_labels.add(label)

        if duplicate_labels:
            raise ValueError(
                f"Worksheet {sheet_name!r} contains duplicate column names: "
                f"{duplicate_labels}."
            )

        # 5. Read the worksheet into a DataFrame.
        data = pd.read_excel(workbook, sheet_name=sheet_name)

        # 6. Check that the DataFrame still has no duplicate column names.
        duplicated_columns = data.columns[data.columns.duplicated()].tolist()
        if duplicated_columns:
            raise ValueError(
                f"Worksheet {sheet_name!r} contains duplicate column names: "
                f"{duplicated_columns}."
            )

        required_columns = list(required_columns)
        missing_columns = [
            column for column in required_columns if column not in data.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Worksheet {sheet_name!r} is missing required columns: "
                f"{missing_columns}."
            )

        # 7. Return the uncleaned and untransformed DataFrame.
        return data
    finally:
        workbook.close()


def extract_implied_volatility_from_excel(excel_file):
    """
    Extract the standardized implied-volatility worksheet.

    Parameters
    ----------
    excel_file : str or pathlib.Path
        Path to an Excel workbook containing the worksheet
        implied_volatility.

    Returns
    -------
    implied_volatility_data : pandas.DataFrame
        Two-dimensional implied-volatility matrix. The index contains the
        maturities, the columns contain moneyness values K / S_0, and the
        entries contain implied_volatility values. The input log-moneyness
        values are converted using moneyness = exp(log_moneyness).

    Raises
    ------
    FileNotFoundError
        If excel_file is not an existing file.

    ValueError
        If the worksheet is absent, if a required column is missing, or if
        duplicate column names are present, or if a pair of maturity and
        log-moneyness observations occurs more than once.

    Notes
    -----
    The worksheet is stored in long format, but the returned DataFrame is
    reshaped for calibration use. This reshaping is not a data-cleaning
    operation: duplicate observations are rejected rather than aggregated.
    """
    # 1. Extract the standardized long-format worksheet.
    data = _extract_sheet(
        excel_file,
        IMPLIED_VOLATILITY_SHEET,
        IMPLIED_VOLATILITY_COLUMNS,
    )

    # 2. Convert log-moneyness into moneyness K / S_0.
    data = data.assign(moneyness=np.exp(data["log_moneyness"]))

    # 3. Reshape the long table into a maturity-by-moneyness matrix.
    return data.pivot(
        index="maturity",
        columns="moneyness",
        values="implied_volatility",
    ).sort_index().sort_index(axis=1)


def extract_variance_swaps_from_excel(excel_file):
    """
    Extract the standardized variance-swaps worksheet.

    Parameters
    ----------
    excel_file : str or pathlib.Path
        Path to an Excel workbook containing the worksheet variance_swaps.

    Returns
    -------
    variance_swaps_data : pandas.DataFrame
        DataFrame read from variance_swaps. The expected columns are maturity
        and variance_swap_quote. The rows and values are not cleaned,
        sorted, or transformed.

    Raises
    ------
    FileNotFoundError
        If excel_file is not an existing file.

    ValueError
        If the worksheet is absent, if a required column is missing, or if
        duplicate column names are present.
    """
    # 1. Extract the standardized variance-swaps worksheet.
    return _extract_sheet(
        excel_file,
        VARIANCE_SWAP_SHEET,
        VARIANCE_SWAP_COLUMNS,
    )


def extract_volatility_swaps_from_excel(excel_file):
    """
    Extract the standardized volatility-swaps worksheet.

    Parameters
    ----------
    excel_file : str or pathlib.Path
        Path to an Excel workbook containing the worksheet volatility_swaps.

    Returns
    -------
    volatility_swaps_data : pandas.DataFrame
        DataFrame read from volatility_swaps. The expected columns are
        maturity and volatility_swap_quote. The rows and values are not
        cleaned, sorted, or transformed.

    Raises
    ------
    FileNotFoundError
        If excel_file is not an existing file.

    ValueError
        If the worksheet is absent, if a required column is missing, or if
        duplicate column names are present.
    """
    # 1. Extract the standardized volatility-swaps worksheet.
    return _extract_sheet(
        excel_file,
        VOLATILITY_SWAP_SHEET,
        VOLATILITY_SWAP_COLUMNS,
    )


def extract_market_data_from_excel(excel_file):
    """
    Extract every recognized market-data worksheet in a workbook.

    Parameters
    ----------
    excel_file : str or pathlib.Path
        Path to an Excel workbook containing one or more recognized
        worksheets.

    Returns
    -------
    market_data : dict[str, pandas.DataFrame]
        Dictionary keyed by recognized worksheet name. Only worksheets among
        implied_volatility, variance_swaps, and volatility_swaps are included.
        The implied_volatility value is reshaped into a maturity-by-moneyness
        matrix; the swap values are the corresponding long-format DataFrames.
        No data-value cleaning, sorting, or calibration is performed; only the
        worksheet structure is checked.

    Raises
    ------
    FileNotFoundError
        If excel_file is not an existing file.

    ValueError
        If no recognized worksheet is present, or if a present recognized
        worksheet is missing required columns or contains duplicate column
        names.

    Notes
    -----
    Unrecognized worksheets are ignored. A workbook may therefore contain
    auxiliary worksheets, provided that at least one recognized worksheet is
    present.
    """
    # 1. Check that the Excel file exists.
    path = Path(excel_file)
    if not path.is_file():
        raise FileNotFoundError(f"Excel file not found: {path}")

    # 2. Define recognized worksheets and their required columns.
    sheet_specs = {
        IMPLIED_VOLATILITY_SHEET: IMPLIED_VOLATILITY_COLUMNS,
        VARIANCE_SWAP_SHEET: VARIANCE_SWAP_COLUMNS,
        VOLATILITY_SWAP_SHEET: VOLATILITY_SWAP_COLUMNS,
    }

    workbook = pd.ExcelFile(path)
    try:
        present_sheets = [
            sheet_name
            for sheet_name in sheet_specs
            if sheet_name in workbook.sheet_names
        ]
    finally:
        workbook.close()

    # 3. Identify recognized worksheets present in the workbook.
    if not present_sheets:
        # 4. Raise an error if no recognized worksheet is present.
        raise ValueError(
            f"No recognized market-data worksheet is present in {path}. "
            f"Expected one of {list(sheet_specs)}."
        )

    # 5. Select the extractor for each recognized worksheet.
    sheet_extractors = {
        IMPLIED_VOLATILITY_SHEET: extract_implied_volatility_from_excel,
        VARIANCE_SWAP_SHEET: extract_variance_swaps_from_excel,
        VOLATILITY_SWAP_SHEET: extract_volatility_swaps_from_excel,
    }

    # 6. Extract every recognized worksheet that is present.
    market_data = {
        sheet_name: sheet_extractors[sheet_name](path)
        for sheet_name in present_sheets
    }

    # 7. Return the extracted DataFrames.
    return market_data
