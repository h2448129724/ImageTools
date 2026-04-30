"""Function selection panel with searchable tree view."""
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QLineEdit,
                                QTreeWidget, QTreeWidgetItem)
from PySide6.QtCore import Signal, Qt

from core.function_registry import (
    get_categories, get_functions_by_category, get_all_functions_flat,
)


class FunctionPanel(QWidget):
    functionSelected = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._all_functions = get_all_functions_flat()  # (key, name, cat_name)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        title = QLabel("功能选择")
        f = title.font()
        f.setBold(True)
        title.setFont(f)
        layout.addWidget(title)

        # Search box
        self.search = QLineEdit()
        self.search.setObjectName("searchBox")
        self.search.setPlaceholderText("搜索功能...")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._on_search)
        layout.addWidget(self.search)

        # Tree widget - no internal scrollbar, rely on parent layout
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(12)
        self.tree.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.tree.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.tree.setSizePolicy(QTreeWidget().sizePolicy().horizontalPolicy(),
                                 QTreeWidget().sizePolicy().verticalPolicy())
        self.tree.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.tree)

        self._populate_tree()

    def _populate_tree(self):
        from PySide6.QtWidgets import QSizePolicy
        self.tree.clear()
        self._cat_items = {}
        self._func_items = {}

        for cat_name in get_categories():
            cat_item = QTreeWidgetItem(self.tree, [cat_name])
            cat_item.setData(0, Qt.UserRole, ("category", cat_name))
            f = cat_item.font(0)
            f.setBold(True)
            cat_item.setFont(0, f)
            self._cat_items[cat_name] = cat_item

            for key, name in get_functions_by_category(cat_name):
                func_item = QTreeWidgetItem(cat_item, [name])
                func_item.setData(0, Qt.UserRole, ("function", key, name))
                self._func_items[key] = func_item

        # Expand only the first category by default; others collapsed
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            item.setExpanded(i == 0)

    def _on_item_clicked(self, item):
        data = item.data(0, Qt.UserRole)
        if not data or data[0] != "function":
            return
        _, key, name = data
        self.functionSelected.emit(key, name)

    def _on_search(self, text):
        low = text.lower().strip()
        if not low:
            # Show all, expand all
            for i in range(self.tree.topLevelItemCount()):
                item = self.tree.topLevelItem(i)
                item.setHidden(False)
                for j in range(item.childCount()):
                    item.child(j).setHidden(False)
            self.tree.expandAll()
            return

        # Filter: hide non-matching, expand categories with matches
        for i in range(self.tree.topLevelItemCount()):
            cat_item = self.tree.topLevelItem(i)
            has_match = False
            for j in range(cat_item.childCount()):
                func_item = cat_item.child(j)
                name = func_item.text(0).lower()
                match = low in name
                func_item.setHidden(not match)
                if match:
                    has_match = True
            cat_item.setHidden(not has_match)
            if has_match:
                cat_item.setExpanded(True)

    def select_function(self, key: str):
        """Programmatically select a function by key."""
        item = self._func_items.get(key)
        if item:
            self.tree.setCurrentItem(item)
            self._on_item_clicked(item)
            # Scroll to item
            self.tree.scrollToItem(item)
