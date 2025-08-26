from polpo.models import ClosestLookupModel, ListLookup


def test_list_lookup():
    model = ListLookup([0, 1])
    assert model.predict((0,)) == 0

    model = ListLookup([0, 1], tar=1)
    assert model.predict((1,)) == 0

    model = ListLookup({0: 1, 1: 0}, tar=0)
    assert model.predict((0,)) == 1


def test_closest_lookup_model():
    model = ClosestLookupModel({0: 0, 9: 1})
    assert model.predict((5,)) == 1
