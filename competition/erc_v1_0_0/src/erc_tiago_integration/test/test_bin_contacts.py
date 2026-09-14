from types import SimpleNamespace

from erc_tiago_integration.solution import Trial


def test_bin_sensor_accepts_only_selected_book_during_placement():
    target='book_col_2_row_5_red'
    node=SimpleNamespace(contact_phase='carry',target_book_token=target,bin_contacts=[])
    def message(book):
        return SimpleNamespace(contacts=[SimpleNamespace(
            collision1=SimpleNamespace(name=book+'::base_link_book_collision'),
            collision2=SimpleNamespace(name='erc_collection_bin::bin_collision'))])
    Trial.on_bin_contacts(node,message(target))
    assert not node.bin_contacts
    node.contact_phase='place'
    Trial.on_bin_contacts(node,message('book_col_1_row_2_red'))
    assert not node.bin_contacts
    Trial.on_bin_contacts(node,message(target))
    assert len(node.bin_contacts)==1
