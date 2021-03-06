from routes import db

# new
import datetime
from pytz import timezone

tz = timezone('EST')


class CompanyDocumentModel(db.Model):
    __tablename__ = 'companydocument'

    title = db.Column(db.String(200), primary_key=True)
    clean_text = db.Column(db.Text)
    classified_sentences = db.Column(db.Text)
    reference_to_search_query = db.Column(db.String(200))
    industry_tags = db.Column(db.String(200))
    query_score = db.Column(db.String(200))
    run_query_score = db.Column(db.Boolean)
    # new
    date_created = db.Column(db.DateTime(), index=True, default=datetime.datetime.now(tz))

    @classmethod
    def get_rows(cls):
        return cls.query.all()

    @classmethod
    def get_row_by_title(cls, title):
        return cls.query.filter_by(title=title).first()

    @classmethod
    def delete(cls, title):
        try:
            cls.query.filter_by(title=title).delete()
            db.session.commit()
            return True
        except:
            return False


class SearchQueryDocumentModel(db.Model):
    __tablename__ = 'searchquerydocument'
    title = db.Column(db.String(255))
    clean_text = db.Column(db.Text)
    classified_sentences = db.Column(db.Text)
    author = db.Column(db.String(255))
    provider = db.Column(db.String(255))
    url = db.Column(db.String(255), primary_key=True)
    image_url = db.Column(db.Text)
    date = db.Column(db.String(255))
    f_title = db.Column(db.String(255), primary_key=True)
    # new
    date_created = db.Column(db.DateTime(), index=True, default=datetime.datetime.now(tz))

    @classmethod
    def deleteall(cls, title):
        try:
            cls.query.filter_by(f_title=title).delete()
            db.session.commit()
            return True
        except:
            return False

    @classmethod
    def delete(cls, title):
        try:
            cls.query.filter_by(title=title).delete()
            db.session.commit()
            return True
        except:
            return False


class ArbitraryDocumentModel(db.Model):
    __tablename__ = 'arbitrarydocument'

    title = db.Column(db.String(200), primary_key=True)
    clean_text = db.Column(db.Text)
    classified_sentences = db.Column(db.Text)
    author = db.Column(db.String(200))
    provider = db.Column(db.String(200))
    url = db.Column(db.Text)
    image_url = db.Column(db.Text)
    date = db.Column(db.String(200))
    industry_tags = db.Column(db.String(250))
    # new
    date_created = db.Column(db.DateTime(), index=True, default=datetime.datetime.now(tz))

    @classmethod
    def delete(cls, title):
        try:
            cls.query.filter_by(title=title).delete()
            db.session.commit()
            return True
        except:
            return False


class SearchQueryModel(db.Model):
    __tablename__ = 'searchquery'
    title = db.Column(db.String(200), primary_key=True)
    query_string = db.Column(db.String(200))
    market_language_code = db.Column(db.String(20))
    country_code = db.Column(db.String(20))
    site_type = db.Column(db.String(20))
    site = db.Column(db.String(100))
    characters = db.Column(db.String(20))
    freshness = db.Column(db.String(20))
    fetch_frequency = db.Column(db.String(20))
    status = db.Column(db.String(20))

    date_created = db.Column(db.DateTime(), index=True, default=datetime.datetime.now(tz))

    @classmethod
    def get_rows(cls):
        return cls.query.all()

    @classmethod
    def get_row_by_title(cls, title):
        return cls.query.filter_by(title=title).first()

    @classmethod
    def delete(cls, title):
        try:
            cls.query.filter_by(title=title).delete()
            db.session.commit()
            return True
        except:
            return False


class ReportModel(db.Model):
    __tablename__ = 'report'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200))
    type = db.Column(db.String(50))
    first = db.Column(db.String(100))
    second = db.Column(db.String(100))
    frequency = db.Column(db.String(20))
    up_to_date = db.Column(db.Boolean)
    score = db.Column(db.Text)
    providers = db.Column(db.String(200))
    range_from = db.Column(db.Integer)
    range_to = db.Column(db.Integer)
    dimension = db.Column(db.String(30))
    descending = db.Column(db.Boolean)
    status = db.Column(db.String(20))
    authors = db.Column(db.String(200))
    # new
    date_created = db.Column(db.DateTime(), index=True, default=datetime.datetime.now(tz))

    @classmethod
    def delete(cls, id):
        try:
            cls.query.filter_by(id=id).delete()
            db.session.commit()
            return True
        except:
            return False


class SentenceModel(db.Model):
    __tablename__ = 'sentences'
    id = db.Column(db.Integer, primary_key=True)
    f_id = db.Column(db.Integer)
    sentence1 = db.Column(db.Text)
    similarity = db.Column(db.Integer)
    sentence2 = db.Column(db.Text)
    title = db.Column(db.String(255))
    type = db.Column(db.String(50))
    dimension = db.Column(db.String(30))
    provider = db.Column(db.String(250))
    author = db.Column(db.Text)

    @classmethod
    def delete(cls, f_id):
        try:
            cls.query.filter_by(f_id=f_id).delete()
            db.session.commit()
            return True
        except:
            return False


class IndustryTags(db.Model):
    __tablename__ = 'industrytags'
    title = db.Column(db.String(200), primary_key=True)

    @classmethod
    def delete(cls, title):
        try:
            cls.query.filter_by(title=title).delete()
            db.session.commit()
            return True
        except:
            return False


class Threshold(db.Model):
    __tablename__ = 'threshold'
    id = db.Column(db.Integer, primary_key=True)
    value = db.Column(db.Integer)

class NewDocumentModel(db.Model):
    __tablename__ = 'newdocument'
    id = db.Column(db.Integer, primary_key=True,autoincrement=True)
    author = db.Column(db.String(250))
    text = db.Column(db.Text)
    url = db.Column(db.Text)
    site = db.Column(db.String(250))
    title = db.Column(db.String(255))
    title_full = db.Column(db.String(255))
    published = db.Column(db.String(255))
    replies_count = db.Column(db.Integer)
    participants_count = db.Column(db.Integer)
    site_type = db.Column(db.String(30))
    country = db.Column(db.String(30))
    spam_score = db.Column(db.Float)
    main_image = db.Column(db.Text)
    performance_score = db.Column(db.Float)
    domain_rank = db.Column(db.Integer)
    reach_per_m = db.Column(db.Float)
    reach_views_per_m = db.Column(db.Float)
    reach_views_per_u = db.Column(db.Float)
    reach_updated = db.Column(db.String(100))
    facebook_likes = db.Column(db.Integer)
    facebook_comments = db.Column(db.Integer)
    facebook_shares = db.Column(db.Integer)
    gplus_shares = db.Column(db.Integer)
    pinterest_shares = db.Column(db.Integer)
    linkedin_shares = db.Column(db.Integer)
    stumbledupon_shares = db.Column(db.Integer)
    vk_shares = db.Column(db.Integer)

class NewDocumentPersonsModel(db.Model):
    __tablename__ = 'newdocumentpersons'
    id = db.Column(db.Integer, primary_key=True)
    f_id = db.Column(db.Integer)
    name = db.Column(db.String(100))
    sentiment = db.Column(db.String(100))


class NewDocumentOrganizationsModel(db.Model):
    __tablename__ = 'newdocumentorganizations'
    id = db.Column(db.Integer, primary_key=True)
    f_id = db.Column(db.Integer)
    name = db.Column(db.String(100))
    sentiment = db.Column(db.String(100))


class NewDocumentLocationsModel(db.Model):
    __tablename__ = 'newdocumentlocations'
    id = db.Column(db.Integer, primary_key=True)
    f_id = db.Column(db.Integer)
    name = db.Column(db.String(100))
    sentiment = db.Column(db.String(100))
