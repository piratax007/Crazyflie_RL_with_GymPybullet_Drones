#!/usr/bin/env python3
import argparse
import os
from typing import Optional, List, Sequence, Iterable, Tuple, Callable, Dict
import pandas as pd
from textual.app import App
from textual.widgets import (
    Header, Footer, ListView,
    ListItem, Label, DataTable,
    Static, Input, Button
)
from textual.screen import Screen


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


def ensure_parend_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def export_columns_to_csv(df: pd.DataFrame, indices: List[int], save_path: str) -> str:
    if not indices:
        raise ValueError("No column indices provided")
    selected = df.iloc[:, indices]
    ensure_parend_dir(save_path)
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
            saved = export_columns_to_csv(self.app.df, indices, path)
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


class MenuScreen(Screen):
    def __init__(
            self,
            name: str | None = None,
            id_: str | None = None,
            classes: str | None = None,
    ):
        super().__init__(name, id_, classes)
        self.menu = None
        self.options: List[Tuple[str, str]] = [
            ("columns", "show available data"),
            ("data", "show data"),
            ("export_columns", "export selected columns to CSV")
        ]
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
            "export_columns": ExportColumnsScreen
        }

    def on_mount(self):
        self.set_focus(self.menu)

    def build_menu_items(self) -> List[ListItem]:
        return [ListItem(Label(label), id=key) for key, label in self.options]

    def compose(self):
        yield Header(show_clock=False)
        self.menu = ListView(*self.build_menu_items())
        yield self.menu
        yield Static("Use ↑/↓ and Enter.")
        yield Footer()

    def on_list_view_selected(self, event):
        key = event.item.id
        self.app.push_screen(self.screens[key]())


class TimeseriesPlotter(App):
    CSS = "ListView { height: 1fr; } DataTable { height: 1fr; }"
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()
        self.df = df

    def action_quit(self):
        self.exit()

    def on_mount(self):
        self.push_screen(MenuScreen())


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


# ToDo: A menu entry for given a path and a set of index, export the columns with that index from the original CSV to
#  a new CSV file that should be saved into the given path. If the give path does not exist, create it. The name of
#  the new file should be given by the user as part of the path to save the new file.
# ToDo: A sub-menu "Preprocess the data" with the following options:
#     - ToDo: A file explorer to select the CSV file to be processed
#     - ToDo: Remove NaN values
#     - ToDo: Correct the time stamps
#       - ToDo: Correct the time stamps (e.g. from nanoseconds to seconds)
#       - ToDo: Remove the offset (starting from 0)
#     - ToDo: Remove offset from the data given an offset
#     - ToDo: Cut the first N seconds of the data (time stamps and data)
#     - ToDo: A checkbox to delete the headers row
