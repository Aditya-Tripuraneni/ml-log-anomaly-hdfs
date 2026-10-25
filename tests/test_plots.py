from hdfs_log_anomaly.plots import plot_confusion_matrix

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def texts_of(fig):
    return [t.get_text() for ax in fig.axes for t in ax.texts]


def test_confusion_matrix_shows_counts_and_row_percentages():
    y_true = [0, 0, 0, 1]
    y_pred = [0, 0, 1, 1]
    fig = plot_confusion_matrix(y_true, y_pred, title="demo")
    # Rows are the actual class, so percentages are shares of that class.
    assert texts_of(fig) == ["2\n66.67%", "1\n33.33%", "0\n0.00%", "1\n100.00%"]


def test_confusion_matrix_saves_a_png(tmp_path):
    path = tmp_path / "cm.png"
    plot_confusion_matrix([0, 1, 1, 0], [0, 1, 0, 0], title="demo", path=path)
    assert path.read_bytes().startswith(PNG_SIGNATURE)
