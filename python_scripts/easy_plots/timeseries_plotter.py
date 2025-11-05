#!/usr/bin/env python3
import argparse
import os
from typing import Optional, List, Sequence, Tuple, Callable, Dict
import pandas as pd
from textual.app import App, ComposeResult
from textual.widgets import (
    Header, Footer, ListView,
    ListItem, Label, DataTable,
    Static, Input, Button
)
from textual.screen import Screen
import hashlib


def get_file(file_path: str) -> str:
    if os.path.isfile(file_path):
        return file_path

    raise FileNotFoundError(f"[ERROR]: no file under the specified path: {file_path}")


def load_csv_from(file_path: str, header: Optional[int]=None) -> pd.DataFrame:
    try:
        return pd.read_csv(get_file(file_path), header=header)
    except Exception as e:
        raise RuntimeError(f"[ERROR]: Failed to load CSV {e}") from e


def get_column_names(df: pd.DataFrame) -> List[str]:
    return list(map(str, df.columns.tolist()))


def enumerate_column_headers(names: List[str]) -> List[Tuple[str, str]]:
    return [(str(index), str(name)) for index, name in enumerate(names)]


def dataframe_to_rows(df: pd.DataFrame) -> List[List[str]]:
    return [list(map(str, row)) for row in df.itertuples(index=False, name=None)]


def parse_index_spec(spec: str) -> List[int]:
    tokens = [t.strip() for t in spec.split(',') if t.strip()]
    def token_to_indices(t: str) -> List[int]:
        if '-' in t:
            start, end = map(int, t.split('-', 1))
            step = 1 if end >= start else -1
            return list(range(start, end + step, step))
        return [int(t)]
    return sorted({i for t in tokens for i in token_to_indices(t)})


def ensure_parent_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def export_columns_to_csv(df: pd.DataFrame, indices: List[int], save_path: str) -> str:
    if not indices:
        raise ValueError("No column indices provided")
    selected = df.iloc[:, indices]
    ensure_parent_dir(save_path)
    selected.to_csv(save_path, index=False)
    return save_path


class ExportColumnsScreen(Screen):
    BINDINGS = [("escape", "pop_screen", "Back to menu"), ("b", "pop_screen", "Back to menu")]

    def __init__(
            self,
            name: str | None = None,
            id_: str | None = None,
            classes: str | None = None,
    ):
        super().__init__(name, id_, classes)
        self.status = Static("Enter a file path and column indices (e.g., 0,2-4,7)")
        self.path_input = Input(placeholder="/path/to/output.csv", id="path")
        self.indices_input = Input(placeholder="0,2-4,7", id="indices")
        self.save_button = Button(label="Save", name="save")

    def action_pop_screen(self):
        self.app.pop_screen()

    def compose(self):
        yield Header(show_clock=False)
        yield self.status
        yield self.path_input
        yield self.indices_input
        yield self.save_button
        yield Footer()

    def on_button_pressed(self, event):
        if event.button.name == "save":
            self._do_save()

    def _do_save(self) -> None:
        try:
            path = self.path_input.value.strip()
            indices = parse_index_spec(self.indices_input.value)
            saved = export_columns_to_csv(self.app.df, indices, path) # type: ignore[attr-defined]
            self.status.update(f"Successfully saved to {saved}")
        except Exception as e:
            self.status.update(f"[Error]: {e}")


def add_content_to_table(
        table: DataTable,
        headers: List[str],
        rows: List[Sequence[str]],
        title: str | None = None
) -> None:
    table.clear(columns=True)
    table.add_columns(*map(str, headers))
    _ = list(map(lambda row: table.add_row(*map(str, row)), rows))

    if title is not None:
        table.title = title


def table_screen_fabricator(
        title: str,
        headers_fn: Callable[[pd.DataFrame], List[str]],
        rows_fn: Callable[[pd.DataFrame], List[Sequence[str]]],
) -> type[Screen]:
    class TableScreen(Screen):
        BINDINGS = [("escape", "pop_screen", "Back to menu"), ("b", "pop_screen", "Back to menu")]

        def action_pop_screen(self):
            self.app.pop_screen()

        def compose(self):
            yield Header(show_clock=False)
            table = DataTable(zebra_stripes=True)
            table.cursor_type = "row"
            headers = headers_fn(self.app.df) # type: ignore[attr-defined]
            rows = rows_fn(self.app.df) # type: ignore[attr-defined]
            add_content_to_table(table, headers, rows, title=title)
            yield table
            yield Static("Press Esc or b to return to menu.")
            yield Footer()

    return TableScreen


def build_menu_items(options: List[Tuple[str, str]]) -> List[ListItem]:
    return [ListItem(Label(label), id=key) for key, label in options]


def common_menu_screen_content():
    yield Header(show_clock=False)
    yield Static("Use ↑/↓ and Enter.")
    yield Footer()


class MainMenuScreen(Screen):
    def __init__(
            self,
            name: str | None = None,
            id_: str | None = None,
            classes: str | None = None,
    ):
        super().__init__(name, id_, classes)
        self.options: List[Tuple[str, str]] = [
            ("columns", "show available data"),
            ("data", "show data"),
            ("export_columns", "export selected columns to CSV"),
            ("preprocess", "preprocess data")
        ]
        self.menu = ListView(*build_menu_items(self.options))
        self.screens: Dict[str, type[Screen]] = {
            "columns": table_screen_fabricator(
                title = "CSV Column Headers",
                headers_fn = lambda df: ["Index", "Column"],
                rows_fn = lambda df: enumerate_column_headers(get_column_names(df))
            ),
            "data": table_screen_fabricator(
                title = "CSV Data",
                headers_fn = get_column_names,
                rows_fn = dataframe_to_rows
            ),
            "export_columns": ExportColumnsScreen,
            "preprocess": PreprocessDataMenuScreen
        }

    def compose(self) -> ComposeResult:
        yield self.menu
        yield from common_menu_screen_content()

    def on_mount(self):
        self.set_focus(self.menu)

    def on_list_view_selected(self, event):
        key = event.item.id
        screen_cls = self.screens.get(key)
        if screen_cls is not None:
            self.app.push_screen(self.screens[key]())


class FileExplorerScreen(Screen):
    BINDINGS = [("escape", "pop_screen", "Back to menu"), ("b", "pop_screen", "Back to menu")]

    def __init__(self, start_path: Optional[str] = None):
        super().__init__()
        self.current_working_directory = os.path.abspath(start_path or os.getcwd())
        self.path_label = Static("")
        self.csv_file_list: Optional[ListView] = None
        self.status = Static("Navigate with Enter. Choose a .csv file to load.")
        self.path_by_id: Dict[str, str] = {}

    def action_pop_screen(self):
        self.app.pop_screen()

    def compose(self):
        yield Header(show_clock=False)
        self.path_label.update(self.current_working_directory)
        yield self.path_label
        self.csv_file_list = ListView()
        yield self.csv_file_list
        yield self.status
        yield Footer()

    @staticmethod
    def _get_entries(path: str) -> List[Tuple[str, str]]:
        entries = []
        parent = os.path.dirname(path)
        entries.append(("[..]", parent))

        try:
            with os.scandir(path) as it0:
                dirs = sorted([e.name for e in it0 if e.is_dir() and not e.name.startswith(".")])
            with os.scandir(path) as it1:
                files = sorted([e.name for e in it1 if e.is_file() and e.name.lower().endswith('.csv')])
        except PermissionError:
            dirs, files = [], []

        entries.extend([(f"[d] {d}", os.path.join(path, d)) for d in dirs])
        entries.extend([(f"[f] {f}", os.path.join(path, f)) for f in files])
        return entries

    def on_list_view_selected(self, event):
        key = event.item.id
        target = self.path_by_id.get(key)

        if not target:
            self.status.update(f"[Error]: Unknown selection {key}")
            return

        if os.path.isdir(target):
            self.current_working_directory = os.path.abspath(target)
            self._refresh_csv_file_list()
            return

        try:
            self.app.df = load_csv_from(target, header=self.app.header) # type: ignore[attr-defined]
            self.app.flash_message = f"Successfully loaded {target}"
            self.app.pop_screen()
        except Exception as e:
            self.status.update(f"[Error]: {e}")

    def _generate_unique_key(self, full_path: str) -> str:
        digest = hashlib.sha1(os.path.abspath(full_path).encode()).hexdigest()[:16]
        base_key = f"f{digest}"

        final_key = base_key
        suffix = 0
        while final_key in self.path_by_id:
            suffix += 1
            final_key = f"{base_key}_{suffix}"
        return final_key

    def _update_file_list(self, entries: list[tuple[str, str]]) -> None:
        assert self.csv_file_list is not None
        self.csv_file_list.clear()
        self.path_by_id.clear()

        for label, full_path in entries:
            unique_key = self._generate_unique_key(full_path)
            self.path_by_id[unique_key] = full_path
            self.csv_file_list.append(ListItem(Label(label), id=unique_key))

        self.path_label.update(self.current_working_directory)

    def _refresh_csv_file_list(self) -> None:
        entries = self._get_entries(self.current_working_directory)
        self._update_file_list(entries)

    def on_mount(self):
        self._refresh_csv_file_list()
        self.set_focus(self.csv_file_list)


class RemoveNaNScreen(Screen):
    BINDINGS = [("escape", "pop_screen", "Back to menu"), ("b", "pop_screen", "Back to menu")]

    def __init__(self):
        super().__init__()
        self.summary = Static("")
        self.remove_button = Button(label="Remove NaN rows", name="remove")
        self.note = Static("This will drop any row that contains at least one NaN value.")

    def action_pop_screen(self):
        self.app.pop_screen()

    def compose(self):
        yield Header(show_clock=False)
        yield self.summary
        yield self.note
        yield self.remove_button
        yield Footer()

    def _count_rows_with_nans(self) -> int:
        return int(self.app.df.isna().any(axis=1).sum()) # type: ignore[attr-defined]

    def _counting_rows(self) -> Tuple[int, int, int]:
        df = self.app.df # type: ignore[attr-defined]
        count_rows = len(df)
        number_of_rows_with_nans = self._count_rows_with_nans() # type: ignore[attr-defined]
        number_of_rows_without_nans = count_rows - number_of_rows_with_nans
        return count_rows, number_of_rows_with_nans, number_of_rows_without_nans

    def _update_summary(self) -> None:
        t, nans, non_nans = self._counting_rows()
        self.summary.update(
            f"Rows: total={t}, with NaNs={nans}, without NaNs={non_nans}"
        )

    def on_mount(self):
        self._update_summary()

    def _drop_rows_with_nans(self) -> pd.DataFrame:
        df = self.app.df # type: ignore[attr-defined]
        return df.dropna(axis=0, how="any").reset_index(drop=True)

    def on_button_pressed(self, event):
        if event.button.name == "remove":
            t, nans, non_nans = self._counting_rows()
            self.app.df = self._drop_rows_with_nans() # type: ignore[attr-defined]
            self.app.flash_message = f"Removed {nans} rows with NaN values (from {t} total rows to {non_nans} rows without NaNs).)"
            self.app.pop_screen()


class PreprocessDataMenuScreen(Screen):
    def __init__(self):
        super().__init__()
        self.options: List[Tuple[str, str]] = [
            ("select_csv", "select CSV file"),
            ("remove_nans", "remove rows containing NaN values")
        ]
        self.menu = ListView(*build_menu_items(self.options))
        self.status = Static("")
        self.screens: Dict[str, type[Screen]] = {
            "select_csv": FileExplorerScreen,
            "remove_nans": RemoveNaNScreen
        }

    BINDINGS = [("escape", "pop_screen", "Back to menu"), ("b", "pop_screen", "Back to menu")]

    def action_pop_screen(self):
        self.app.pop_screen()

    def compose(self):
        yield self.menu
        yield self.status
        yield from common_menu_screen_content()

    def _consume_app_flash_message(self) -> None:
        msg = getattr(self.app, "flash_message", None)
        if msg:
            self.status.update(msg)
            self.app.flash_message = None

    def on_mount(self):
        self._consume_app_flash_message()
        self.set_focus(self.menu)

    def _on_screen_resume(self):
        self._consume_app_flash_message()

    def on_list_view_selected(self, event):
        key = event.item.id
        self.app.push_screen(self.screens[key]())


class TimeseriesPlotter(App):
    CSS = "ListView { height: 1fr; } DataTable { height: 1fr; }"
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()
        self.df = df
        self.header: Optional[int] = None
        self.flash_message: Optional[str] = None

    def action_quit(self):
        self.exit()

    def on_mount(self):
        self.push_screen(MainMenuScreen())


def timeseries_plotter_runner(df: pd.DataFrame) -> None:
    TimeseriesPlotter(df).run()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CSV menu app for rosbags")
    parser.add_argument("--file_path", required=True, help="Path to the CSV file")
    parser.add_argument(
        "--header",
        default=0,
        type=lambda x: None if x in ("None", "none", "") else int(x),
        help="Header row index for pandas (default: 0). Use None if there is no header.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    df = load_csv_from(args.file_path, header=args.header)
    timeseries_plotter_runner(df)


if __name__ == "__main__":
    main()


# 1. Done: A menu entry for given a path and a set of indices, export the columns with that indices from the original
# CSV to a new CSV file that should be saved into the given path. If the give path does not exist, create it.
# The name of the new file should be given by the user as part of the path to save the new file.
# 2. ToDo: A sub-menu "Preprocess the data" with the following options:
#     - 2.1 Done: A file explorer to select the CSV file to be processed
#     - 2.2 Done: Remove NaN values by removing the whole row containing NaN values
#     - 2.3 ToDo: Correct the time stamps
#       - 2.3.1 ToDo: Correct the time stamps (e.g. from nanoseconds to seconds)
#       - 2.3.2 ToDo: Remove the offset (starting from 0)
#     - 2.4 ToDo: Remove offset from the data given an offset
#     - 2.5 ToDo: Cut the first N seconds of the data (time stamps and data)
#     - 2.6 ToDo: A checkbox to delete the headers row
#     - 2.7 ToDo: A "save" button to save the processed data to a new CSV file
