from db import engine, Base
import models  # noqa - importing is what registers the tables on Base

Base.metadata.create_all(engine)
print("created:", list(Base.metadata.tables))