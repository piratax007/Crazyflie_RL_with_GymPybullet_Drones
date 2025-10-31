#!/usr/bin/env python3
import argparse
import os
from typing import List, Sequence, Iterable, Tuple, Callable, Dict
import pandas as pd
from textual.app import App
from textual.widgets import Header, Footer, ListView, ListItem, Label, DataTable, Static
from textual.screen import Screen


def get_file(file_path: str) -> str:
    if os.path.isfile(file_path):
        return file_path

    raise FileNotFoundError(f"[ERROR]: no file under the specified path: {file_path}")


def load_csv_from(file_path: str, header: int=None) -> pd.DataFrame:
    return pd.read_csv(get_file(file_path), header=header)


def build_menu_items(options: Iterable[Tuple[str, str]]) -> List[ListItem]:
    return [ListItem(Label(label, id=key)) for key, label in options]


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


def get_column_names(df: pd.DataFrame) -> List[str]:
    return list(map(str, df.columns.tolist()))


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


_MENU_OPTIONS: List[Tuple[str, str]] = [
    ("show_columns", "show available data"),
    ("show_data", "show data")
]


class MenuScreen(Screen):
    def compose(self):
        yield Header(show_clock=False)
        self.menu = ListView(*build_menu_items(_MENU_OPTIONS))
        yield self.menu
        yield Static("Use ↑/↓ and Enter.")
        yield Footer()

    def on_list_view_selected(self, event):
        key = event.item.children[0].id
        if key == "show_columns":
            self.app.push_screen(screens["columns"]())
        elif key == "show_data":
            self.app.push_screen(screens["data"]())


def enumerate_column_headers(names: List[str]) -> List[Tuple[str, str]]:
    return [(str(index), str(name)) for index, name in enumerate(names)]


def dataframe_to_rows(df: pd.DataFrame) -> List[List[str]]:
    return [list(map(str, row)) for row in df.itertuples(index=False, name=None)]


screens: Dict[str, type[Screen]] = {
    "menu": MenuScreen,
    "columns": table_screen_fabricator(
        title = "CSV Column Headers",
        headers_fn = lambda df: ["Index", "Column"],
        rows_fn = lambda df: enumerate_column_headers(get_column_names(df))
    ),
    "data": table_screen_fabricator(
        title = "CSV Data",
        headers_fn = get_column_names,
        rows_fn = dataframe_to_rows
    )
}

class TimeseriesPlotter(App):
    CSS = "ListView { height: 1fr; } DataTable { height: 1fr; }"
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()
        self.df = df

    def action_quit(self):
        self.exit()

    def on_mount(self):
        self.push_screen(screens["menu"]())


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
