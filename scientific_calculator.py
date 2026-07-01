#!/usr/bin/env python3
"""
PySide6 Scientific Calculator for Windows
A modular, safe, and maintainable scientific calculator with:
- Basic and advanced mathematical operations
- Plotting capabilities
- Calculation history
- Variable storage
- Symbolic mathematics (via SymPy)
"""

import sys
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any

# Third-party imports
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QTextEdit, QTabWidget, QGridLayout,
    QLabel, QMessageBox, QFileDialog, QMenu, QMenuBar, QAction,
    QSplitter, QFrame, QScrollArea, QListWidget, QListWidgetItem,
    QDialog, QDialogButtonBox, QFormLayout, QComboBox, QDoubleSpinBox
)
from PySide6.QtCore import Qt, QSize, Signal, Slot
from PySide6.QtGui import QFont, QIcon, QKeySequence, QShortcut

import numpy as np
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import sympy
from sympy import sympify, symbols, diff, integrate, simplify, expand, factor
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application


class CalculatorEngine:
    """Core calculation engine with safe evaluation."""
    
    def __init__(self):
        self.variables: Dict[str, float] = {}
        self.history: list = []
        self.max_history = 100
        
        # Safe math context
        self.safe_dict = {
            'pi': np.pi, 'e': np.e, 'tau': np.tau,
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'asin': np.arcsin, 'acos': np.arccos, 'atan': np.arctan,
            'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
            'log': np.log10, 'log10': np.log10, 'ln': np.log,
            'exp': np.exp, 'sqrt': np.sqrt, 'abs': np.abs,
            'floor': np.floor, 'ceil': np.ceil, 'round': round,
            'deg': np.degrees, 'rad': np.radians,
            'fact': lambda x: np.math.factorial(int(x)),
            'gamma': lambda x: np.math.gamma(x),
        }
        
    def evaluate(self, expression: str) -> tuple[bool, Any]:
        """Safely evaluate a mathematical expression."""
        try:
            expr = expression.strip()
            if not expr:
                return False, "Empty expression"
            
            # Replace common symbols
            expr = expr.replace('^', '**').replace('×', '*').replace('÷', '/')
            
            # Add variables to safe dict
            eval_dict = {**self.safe_dict, **self.variables}
            
            result = eval(compile(expr, '<string>', 'eval'), {"__builtins__": {}}, eval_dict)
            
            # Handle numpy types
            if hasattr(result, 'item'):
                result = float(result.item())
            elif isinstance(result, (np.ndarray, list)):
                result = result.tolist()
                
            self._add_to_history(expression, result)
            return True, result
            
        except Exception as e:
            return False, str(e)
    
    def _add_to_history(self, expression: str, result: Any):
        """Add calculation to history."""
        self.history.append({
            'expression': expression,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def set_variable(self, name: str, value: float) -> bool:
        """Set a variable value."""
        if not name.isidentifier():
            return False
        self.variables[name] = float(value)
        return True
    
    def get_variable(self, name: str) -> Optional[float]:
        """Get a variable value."""
        return self.variables.get(name)
    
    def clear_history(self):
        """Clear calculation history."""
        self.history.clear()
    
    def get_history(self) -> list:
        """Get calculation history."""
        return self.history.copy()


class SymbolicEngine:
    """Symbolic mathematics engine using SymPy."""
    
    def __init__(self):
        self.x, self.y, self.z = symbols('x y z')
        self.transformations = (standard_transformations + (implicit_multiplication_application,))
    
    def simplify_expression(self, expr: str) -> str:
        """Simplify a symbolic expression."""
        try:
            parsed = parse_expr(expr, transformations=self.transformations)
            return str(simplify(parsed))
        except Exception as e:
            return f"Error: {e}"
    
    def differentiate(self, expr: str, var: str = 'x') -> str:
        """Differentiate an expression."""
        try:
            parsed = parse_expr(expr, transformations=self.transformations)
            symbol = symbols(var)
            result = diff(parsed, symbol)
            return str(result)
        except Exception as e:
            return f"Error: {e}"
    
    def integrate_expr(self, expr: str, var: str = 'x') -> str:
        """Integrate an expression."""
        try:
            parsed = parse_expr(expr, transformations=self.transformations)
            symbol = symbols(var)
            result = integrate(parsed, symbol)
            return str(result)
        except Exception as e:
            return f"Error: {e}"
    
    def expand_expression(self, expr: str) -> str:
        """Expand an expression."""
        try:
            parsed = parse_expr(expr, transformations=self.transformations)
            return str(expand(parsed))
        except Exception as e:
            return f"Error: {e}"
    
    def factor_expression(self, expr: str) -> str:
        """Factor an expression."""
        try:
            parsed = parse_expr(expr, transformations=self.transformations)
            return str(factor(parsed))
        except Exception as e:
            return f"Error: {e}"
    
    def solve_equation(self, expr: str, var: str = 'x') -> str:
        """Solve an equation (expr = 0)."""
        try:
            from sympy import solve
            parsed = parse_expr(expr, transformations=self.transformations)
            symbol = symbols(var)
            solutions = solve(parsed, symbol)
            return ', '.join(str(s) for s in solutions)
        except Exception as e:
            return f"Error: {e}"


class PlotWidget(QWidget):
    """Matplotlib plotting widget."""
    
    def __init__(self):
        super().__init__()
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        
        self.axes = self.figure.add_subplot(111)
        self.axes.grid(True, alpha=0.3)
        
    def plot_function(self, expression: str, x_min: float = -10, x_max: float = 10, 
                      num_points: int = 1000) -> bool:
        """Plot a mathematical function."""
        try:
            self.axes.clear()
            self.axes.grid(True, alpha=0.3)
            
            x = np.linspace(x_min, x_max, num_points)
            
            # Create safe evaluation context
            safe_dict = {
                'x': x, 'pi': np.pi, 'e': np.e,
                'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                'asin': np.arcsin, 'acos': np.arccos, 'atan': np.arctan,
                'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
                'log': np.log10, 'log10': np.log10, 'ln': np.log,
                'exp': np.exp, 'sqrt': np.sqrt, 'abs': np.abs,
            }
            
            expr = expression.replace('^', '**')
            y = eval(expr, {"__builtins__": {}}, safe_dict)
            
            self.axes.plot(x, y, linewidth=2)
            self.axes.set_xlabel('x')
            self.axes.set_ylabel('y')
            self.axes.set_title(f'f(x) = {expression}')
            self.axes.axhline(y=0, color='k', linewidth=0.5)
            self.axes.axvline(x=0, color='k', linewidth=0.5)
            
            self.canvas.draw()
            return True
            
        except Exception as e:
            QMessageBox.critical(self, "Plot Error", f"Could not plot: {e}")
            return False
    
    def clear_plot(self):
        """Clear the plot."""
        self.axes.clear()
        self.axes.grid(True, alpha=0.3)
        self.canvas.draw()
    
    def save_plot(self, filename: str) -> bool:
        """Save the plot to a file."""
        try:
            self.figure.savefig(filename, dpi=150, bbox_inches='tight')
            return True
        except Exception as e:
            return False


class HistoryWidget(QListWidget):
    """Widget to display calculation history."""
    
    history_selected = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.setAlternatingRowColors(True)
        self.itemDoubleClicked.connect(self._on_item_double_clicked)
        
    def update_history(self, history: list):
        """Update the history display."""
        self.clear()
        for item in reversed(history):
            expr = item['expression']
            result = item['result']
            display = f"{expr} = {result}"
            list_item = QListWidgetItem(display)
            list_item.setData(Qt.UserRole, expr)
            self.addItem(list_item)
    
    def _on_item_double_clicked(self, item: QListWidgetItem):
        """Emit signal when item is double-clicked."""
        expr = item.data(Qt.UserRole)
        if expr:
            self.history_selected.emit(expr)


class VariablesDialog(QDialog):
    """Dialog for managing variables."""
    
    def __init__(self, variables: Dict[str, float], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Variables")
        self.setMinimumWidth(400)
        
        self.variables = variables.copy()
        
        layout = QVBoxLayout(self)
        
        self.list_widget = QListWidget()
        self._update_list()
        layout.addWidget(self.list_widget)
        
        form_layout = QFormLayout()
        
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Variable name")
        form_layout.addRow("Name:", self.name_input)
        
        self.value_input = QDoubleSpinBox()
        self.value_input.setRange(-1e308, 1e308)
        self.value_input.setValue(0.0)
        form_layout.addRow("Value:", self.value_input)
        
        layout.addLayout(form_layout)
        
        button_layout = QHBoxLayout()
        
        add_btn = QPushButton("Add/Update")
        add_btn.clicked.connect(self._add_variable)
        button_layout.addWidget(add_btn)
        
        delete_btn = QPushButton("Delete")
        delete_btn.clicked.connect(self._delete_variable)
        button_layout.addWidget(delete_btn)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
    
    def _update_list(self):
        """Update the variables list."""
        self.list_widget.clear()
        for name, value in self.variables.items():
            self.list_widget.addItem(f"{name} = {value}")
    
    def _add_variable(self):
        """Add or update a variable."""
        name = self.name_input.text().strip()
        value = self.value_input.value()
        
        if name and name.isidentifier():
            self.variables[name] = value
            self._update_list()
            self.name_input.clear()
        else:
            QMessageBox.warning(self, "Invalid Name", 
                              "Variable name must be a valid identifier.")
    
    def _delete_variable(self):
        """Delete selected variable."""
        current = self.list_widget.currentRow()
        if current >= 0:
            item = self.list_widget.item(current)
            text = item.text()
            name = text.split('=')[0].strip()
            if name in self.variables:
                del self.variables[name]
                self._update_list()
    
    def get_variables(self) -> Dict[str, float]:
        """Get the updated variables."""
        return self.variables.copy()


class CalculatorWindow(QMainWindow):
    """Main calculator window."""
    
    def __init__(self):
        super().__init__()
        
        self.engine = CalculatorEngine()
        self.symbolic_engine = SymbolicEngine()
        
        self.setWindowTitle("Scientific Calculator")
        self.setMinimumSize(900, 700)
        
        self._setup_ui()
        self._setup_menu()
        self._setup_shortcuts()
        
    def _setup_ui(self):
        """Setup the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        # Tab widget for different views
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Calculator tab
        calc_tab = self._create_calculator_tab()
        self.tabs.addTab(calc_tab, "Calculator")
        
        # Plotting tab
        plot_tab = self._create_plot_tab()
        self.tabs.addTab(plot_tab, "Plot")
        
        # Symbolic math tab
        symbolic_tab = self._create_symbolic_tab()
        self.tabs.addTab(symbolic_tab, "Symbolic")
        
        # History tab
        history_tab = self._create_history_tab()
        self.tabs.addTab(history_tab, "History")
    
    def _create_calculator_tab(self) -> QWidget:
        """Create the calculator tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Display
        self.display = QLineEdit()
        self.display.setFont(QFont("Consolas", 16))
        self.display.setAlignment(Qt.AlignRight)
        self.display.setReadOnly(False)
        self.display.setPlaceholderText("Enter expression...")
        self.display.returnPressed.connect(self._calculate)
        layout.addWidget(self.display)
        
        # Result label
        self.result_label = QLabel("Result:")
        self.result_label.setFont(QFont("Consolas", 14))
        self.result_label.setAlignment(Qt.AlignRight)
        layout.addWidget(self.result_label)
        
        # Buttons grid
        buttons_widget = QWidget()
        grid_layout = QGridLayout(buttons_widget)
        
        # Button definitions
        buttons = [
            ['7', '8', '9', '÷', 'C'],
            ['4', '5', '6', '×', '⌫'],
            ['1', '2', '3', '-', '±'],
            ['0', '.', '(', ')', '+'],
            ['sin', 'cos', 'tan', 'log', 'ln'],
            ['asin', 'acos', 'atan', '√', '^'],
            ['pi', 'e', 'exp', '(', ')'],
            ['fact', 'deg', 'rad', '%', '='],
        ]
        
        for row, button_row in enumerate(buttons):
            for col, btn_text in enumerate(button_row):
                btn = QPushButton(btn_text)
                btn.setFont(QFont("Arial", 12))
                btn.setMinimumSize(60, 40)
                btn.clicked.connect(lambda checked, t=btn_text: self._button_click(t))
                grid_layout.addWidget(btn, row, col)
        
        layout.addWidget(buttons_widget)
        
        # Variables button
        vars_btn = QPushButton("Manage Variables")
        vars_btn.clicked.connect(self._manage_variables)
        layout.addWidget(vars_btn)
        
        return widget
    
    def _create_plot_tab(self) -> QWidget:
        """Create the plotting tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Function input
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("f(x) = "))
        self.plot_input = QLineEdit()
        self.plot_input.setPlaceholderText("e.g., sin(x), x^2, exp(-x)")
        self.plot_input.returnPressed.connect(self._plot_function)
        input_layout.addWidget(self.plot_input)
        
        plot_btn = QPushButton("Plot")
        plot_btn.clicked.connect(self._plot_function)
        input_layout.addWidget(plot_btn)
        
        layout.addLayout(input_layout)
        
        # Range inputs
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("X range:"))
        
        self.x_min_input = QDoubleSpinBox()
        self.x_min_input.setRange(-1000, 1000)
        self.x_min_input.setValue(-10)
        range_layout.addWidget(self.x_min_input)
        
        range_layout.addWidget(QLabel("to"))
        
        self.x_max_input = QDoubleSpinBox()
        self.x_max_input.setRange(-1000, 1000)
        self.x_max_input.setValue(10)
        range_layout.addWidget(self.x_max_input)
        
        layout.addLayout(range_layout)
        
        # Plot widget
        self.plot_widget = PlotWidget()
        layout.addWidget(self.plot_widget)
        
        # Save button
        save_btn = QPushButton("Save Plot")
        save_btn.clicked.connect(self._save_plot)
        layout.addWidget(save_btn)
        
        return widget
    
    def _create_symbolic_tab(self) -> QWidget:
        """Create the symbolic mathematics tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Expression input
        self.symbolic_input = QLineEdit()
        self.symbolic_input.setPlaceholderText("Enter expression (e.g., x^2 + 2*x + 1)")
        layout.addWidget(self.symbolic_input)
        
        # Operation buttons
        ops_layout = QGridLayout()
        
        ops = [
            ("Simplify", self._symbolic_simplify),
            ("Expand", self._symbolic_expand),
            ("Factor", self._symbolic_factor),
            ("Differentiate", self._symbolic_diff),
            ("Integrate", self._symbolic_integrate),
            ("Solve = 0", self._symbolic_solve),
        ]
        
        for i, (text, handler) in enumerate(ops):
            btn = QPushButton(text)
            btn.clicked.connect(handler)
            ops_layout.addWidget(btn, i // 3, i % 3)
        
        layout.addLayout(ops_layout)
        
        # Result display
        self.symbolic_result = QTextEdit()
        self.symbolic_result.setReadOnly(True)
        self.symbolic_result.setFont(QFont("Consolas", 11))
        layout.addWidget(self.symbolic_result)
        
        return widget
    
    def _create_history_tab(self) -> QWidget:
        """Create the history tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        self.history_widget = HistoryWidget()
        self.history_widget.history_selected.connect(self._load_from_history)
        layout.addWidget(self.history_widget)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_history)
        btn_layout.addWidget(refresh_btn)
        
        clear_btn = QPushButton("Clear History")
        clear_btn.clicked.connect(self._clear_history)
        btn_layout.addWidget(clear_btn)
        
        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self._export_history)
        btn_layout.addWidget(export_btn)
        
        layout.addLayout(btn_layout)
        
        return widget
    
    def _setup_menu(self):
        """Setup the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        export_action = QAction("Export History", self)
        export_action.setShortcut(QKeySequence("Ctrl+E"))
        export_action.triggered.connect(self._export_history)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("Edit")
        
        clear_action = QAction("Clear Display", self)
        clear_action.setShortcut(QKeySequence("Ctrl+L"))
        clear_action.triggered.connect(lambda: self.display.clear())
        edit_menu.addAction(clear_action)
        
        clear_hist_action = QAction("Clear History", self)
        clear_hist_action.setShortcut(QKeySequence("Ctrl+H"))
        clear_hist_action.triggered.connect(self._clear_history)
        edit_menu.addAction(clear_hist_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        QShortcut(QKeySequence("F5"), self, self._refresh_history)
        QShortcut(QKeySequence("Escape"), self, lambda: self.display.clear())
    
    def _button_click(self, text: str):
        """Handle button clicks."""
        if text == '=' or text == 'Calculate':
            self._calculate()
        elif text == 'C':
            self.display.clear()
            self.result_label.setText("Result:")
        elif text == '⌫':
            self.display.setText(self.display.text()[:-1])
        elif text == '±':
            current = self.display.text()
            if current.startswith('-'):
                self.display.setText(current[1:])
            elif current:
                self.display.setText('-' + current)
        else:
            # Insert button text
            cursor_pos = self.display.cursorPosition()
            text_to_insert = text
            
            # Map special symbols
            symbol_map = {
                '×': '*', '÷': '/', '√': 'sqrt(', '^': '**',
                'pi': 'pi', 'e': 'e',
            }
            text_to_insert = symbol_map.get(text, text)
            
            # Add parentheses for functions
            if text in ['sin', 'cos', 'tan', 'asin', 'acos', 'atan', 
                       'log', 'ln', 'exp', 'sqrt', 'fact']:
                text_to_insert += '('
            
            self.display.insert(text_to_insert)
    
    def _calculate(self):
        """Perform calculation."""
        expression = self.display.text()
        success, result = self.engine.evaluate(expression)
        
        if success:
            self.result_label.setText(f"Result: {result}")
            self._refresh_history()
        else:
            self.result_label.setText(f"Error: {result}")
            QMessageBox.warning(self, "Calculation Error", result)
    
    def _plot_function(self):
        """Plot the entered function."""
        expression = self.plot_input.text()
        x_min = self.x_min_input.value()
        x_max = self.x_max_input.value()
        
        if expression:
            self.plot_widget.plot_function(expression, x_min, x_max)
    
    def _save_plot(self):
        """Save the current plot."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", "", "PNG Files (*.png);;All Files (*)"
        )
        if filename:
            if not filename.endswith('.png'):
                filename += '.png'
            if self.plot_widget.save_plot(filename):
                QMessageBox.information(self, "Success", "Plot saved successfully!")
            else:
                QMessageBox.critical(self, "Error", "Failed to save plot.")
    
    def _symbolic_simplify(self):
        """Simplify the symbolic expression."""
        expr = self.symbolic_input.text()
        result = self.symbolic_engine.simplify_expression(expr)
        self.symbolic_result.setText(f"Simplified:\n{result}")
    
    def _symbolic_expand(self):
        """Expand the symbolic expression."""
        expr = self.symbolic_input.text()
        result = self.symbolic_engine.expand_expression(expr)
        self.symbolic_result.setText(f"Expanded:\n{result}")
    
    def _symbolic_factor(self):
        """Factor the symbolic expression."""
        expr = self.symbolic_input.text()
        result = self.symbolic_engine.factor_expression(expr)
        self.symbolic_result.setText(f"Factored:\n{result}")
    
    def _symbolic_diff(self):
        """Differentiate the symbolic expression."""
        expr = self.symbolic_input.text()
        result = self.symbolic_engine.differentiate(expr)
        self.symbolic_result.setText(f"Derivative (d/dx):\n{result}")
    
    def _symbolic_integrate(self):
        """Integrate the symbolic expression."""
        expr = self.symbolic_input.text()
        result = self.symbolic_engine.integrate_expr(expr)
        self.symbolic_result.setText(f"Integral (∫dx):\n{result} + C")
    
    def _symbolic_solve(self):
        """Solve the symbolic equation."""
        expr = self.symbolic_input.text()
        result = self.symbolic_engine.solve_equation(expr)
        self.symbolic_result.setText(f"Solutions to {expr} = 0:\n{result}")
    
    def _manage_variables(self):
        """Open variables management dialog."""
        dialog = VariablesDialog(self.engine.variables, self)
        if dialog.exec() == QDialog.Accepted:
            self.engine.variables = dialog.get_variables()
    
    def _refresh_history(self):
        """Refresh the history display."""
        history = self.engine.get_history()
        self.history_widget.update_history(history)
    
    def _clear_history(self):
        """Clear calculation history."""
        self.engine.clear_history()
        self._refresh_history()
        QMessageBox.information(self, "History Cleared", "Calculation history has been cleared.")
    
    def _export_history(self):
        """Export history to file."""
        history = self.engine.get_history()
        if not history:
            QMessageBox.information(self, "No History", "No history to export.")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export History", "history.json", "JSON Files (*.json);;All Files (*)"
        )
        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(history, f, indent=2)
                QMessageBox.information(self, "Success", f"History exported to {filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export: {e}")
    
    def _load_from_history(self, expression: str):
        """Load expression from history."""
        self.display.setText(expression)
        self.tabs.setCurrentIndex(0)  # Switch to calculator tab
    
    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Scientific Calculator",
            "<h2>Scientific Calculator</h2>"
            "<p>A feature-rich scientific calculator built with PySide6.</p>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li>Basic and advanced mathematical operations</li>"
            "<li>Function plotting</li>"
            "<li>Symbolic mathematics (SymPy)</li>"
            "<li>Calculation history</li>"
            "<li>Variable storage</li>"
            "</ul>"
            "<p>Version 1.0.0</p>"
        )


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Set application metadata
    app.setApplicationName("Scientific Calculator")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("SciCalc")
    
    window = CalculatorWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
