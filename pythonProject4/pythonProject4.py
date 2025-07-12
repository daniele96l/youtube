"""Welcome to Reflex! This file outlines the steps to create a basic app."""

import reflex as rx

from rxconfig import config

def dropdown_menu_example():
    return rx.menu.root(
    rx.menu.trigger(
        rx.button("Options", variant="soft"),
    ),
    rx.menu.content(
        rx.menu.item("Edit", shortcut="⌘ E"),
        rx.menu.item("Duplicate", shortcut="⌘ D"),
        rx.menu.separator(),
        rx.menu.item("Archive", shortcut="⌘ N"),
        rx.menu.sub(
            rx.menu.sub_trigger("More"),
            rx.menu.sub_content(
                rx.menu.item("Move to project…"),
                rx.menu.item("Move to folder…"),
                rx.menu.separator(),
                rx.menu.item("Advanced options…"),
            ),
        ),
        rx.menu.separator(),
        rx.menu.item("Share"),
        rx.menu.item("Add to favorites"),
        rx.menu.separator(),
        rx.menu.item("Delete", shortcut="⌘ ⌫", color="red"),
    ),
)
class State(rx.State):
    """The app state."""

def index() -> rx.Component:
    # Welcome Page (Index)
    container = rx.container(
        rx.color_mode.button(position="top-right"),
        rx.vstack(
            rx.heading("Welcome to Reflex!", size="9"),
            rx.text(
                "Get started by editing ",
                rx.code(f"{config.app_name}/{config.app_name}.py"),
                size="5",
            ),
            rx.link(
                rx.button("Check out our docs!"),
                href="https://reflex.dev/docs/getting-started/introduction/",
                is_external=True,
            ),
            spacing="5",
            justify="center",
            min_height="85vh",
        ),
        rx.logo(),
        dropdown_menu_example()
    )
    return container


app = rx.App()
app.add_page(index)
