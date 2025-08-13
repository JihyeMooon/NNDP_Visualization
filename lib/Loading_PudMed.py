# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 00:25:50 2020

@author: Jihye Moon
(jihye.moon@uconn.edu)

"""

from Bio import Entrez
from datetime import datetime
from io import StringIO
import Medline
import os

date = startTime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

class ids_pudmed():
    def __init__(self, snp_ids=[]):
        self.snp_ids=snp_ids
        self.uids=[]
        self.gene_names=[]
        self.names=[]
        self.records=[]
        self.gene_full_names=[]
        self.saved_snp_id=[]
        
    def search_ids(self, search_email):
        removal_index=[]
        Entrez.email = search_email
        records=[]
        for snp_id in self.snp_ids:
            record = Entrez.read(Entrez.elink(dbfrom="snp", 
                                  id=snp_id.replace('rs',''), 
                                  db="gene")) 
            if record[0]['LinkSetDb']==[]:
                removal_index.append(snp_id)
                print("index is removed: ", snp_id)
                
            else:
                results = record[0]['LinkSetDb'][0]['Link']
                multi_gene=[]
                multi_full_name=[]
                multi_uid=[]
                #records=[]
                for result in results:
                    uid = result['Id']
                    handle = Entrez.esummary(db="gene", id=uid)
                    uid_record = Entrez.read(handle)
                    
                    records.append(uid_record)
                    handle.close()
                    uid_summary = uid_record["DocumentSummarySet"]['DocumentSummary'][0]
                    gene_name = uid_summary['Name']
                    gene_full_name = uid_summary['Description']
                    if len(results)>1:
                        multi_gene.append(gene_name)
                        multi_full_name.append(gene_full_name)
                        multi_uid.append(uid)
                        
                        #records.append(uid_record)
                    else:
                        multi_gene = gene_name
                        multi_full_name = gene_full_name
                        multi_uid = uid
                        #records = uid_record
            
                #print(results)
            
                if len(results)>1:
                    multi_uid= "#".join(multi_uid)
                    multi_gene= "#".join(multi_gene) 
                    multi_full_name= "#".join(multi_full_name) 
                    #records= " ".join(records) 
                
                #print(count, "/",len(self.snp_ids)," : ", snp_id, multi_uid, multi_gene)
                self.uids.append(multi_uid)
                self.gene_names.append(multi_gene)
                self.gene_full_names.append(multi_full_name)
                self.saved_snp_id.append(snp_id)
                #self.records.append(records) 
        return removal_index, self.records, self.uids, self.gene_names, self.gene_full_names
        #return records
    def search_id2summary(self, uids, search_email): 
        Entrez.email = search_email
        records=''
        for uid in uids: 
            summary='#'
            handle = Entrez.esummary(db="gene", id=uid)
            #uid_record = Entrez.read(handle) 
            uid_record = Entrez.read(handle,validate=False)
            #records.append(uid_record)
            handle.close()
            #print( uid_record["DocumentSummarySet"]['DocumentSummary'])
            if uid_record["DocumentSummarySet"]['DocumentSummary']==[]:    
                handle = Entrez.esummary(db="gene", id=uid)
                uid_record = Entrez.read(handle) 
                handle.close()
                uid_summary = uid_record["DocumentSummarySet"]['DocumentSummary'][0]
            else:
                uid_summary = uid_record["DocumentSummarySet"]['DocumentSummary'][0]
            gene_name = uid_summary['Name']
            gene_full_name = uid_summary['Description']
            if 'Summary' in uid_summary:
                summary = uid_summary['Summary']
                if summary == '':
                    summary = '.'
            sentence = uid + '\t' + gene_name + '\t' + gene_full_name + '\t' + summary
            records += sentence + '\n'
        return records

    def search_gene2doc(self, query, email):
        LR=[]; TI=[]; AB=[]; MH=[]; RN=[]; PMID=[]; DCOM=[]
        rec_handler = self.search_medline(query, email)

        FullText=''; Meta=''
        for rec_id in rec_handler['IdList']:
            rec = self.fetch_rec(rec_id, rec_handler)
            rec_file = StringIO(rec)
            medline_rec = Medline.read(rec_file)  
            if medline_rec != []:
                if 'LR' in medline_rec:
                    LR.append(medline_rec['LR'])  
                else:
                    LR.append('.')
                if 'TI' in medline_rec:
                    TI.append(medline_rec['TI']) 
                else:
                    TI.append('.')
                if 'AB' in medline_rec:
                    AB.append(medline_rec['AB']) 
                else:
                    AB.append('.')
                if 'MH' in medline_rec:
                    MH.append(medline_rec['MH']) 
                else:
                    MH.append('.')
                if 'PMID' in medline_rec:
                    PMID.append(medline_rec['PMID']) 
                else:
                    PMID.append('.') 
                if 'DCOM' in medline_rec:
                    DCOM.append(medline_rec['DCOM']) 
                else:
                    DCOM.append('.') 
                if 'RN' in medline_rec:
                    RN.append(medline_rec['RN']) 
                else:
                    RN.append('.') 
        for i in range(len(AB)):
            FullText += '\t#'+PMID[i]+'\t'+DCOM[i]+'\t'+LR[i]+'\t'+TI[i]+'\t'+AB[i] 
            Meta += "\t@".join(RN[i])+'\t#'.join(MH[i])
        FullText+='\n' 
        Meta+='\n' 
        return AB, FullText, Meta
     
    def search_medline(self, query, email):
        Entrez.email = email
        search = Entrez.esearch(db='pubmed', term=query, usehistory='y')
        
        handle = Entrez.read(search)
        try:
            return handle
        except Exception as e:
            raise IOError(str(e))
        finally:
            search.close() 

    def search_list(self, query, year, email): 
        self.user_term = query
        self.email = email
        self.year=year
        self.user_db="pubmed"
        
        Entrez.email = email
        if year==None:
            search_results = Entrez.read(
                Entrez.esearch(
                    db=self.user_db, term=self.user_term, datetype="pdat", usehistory="y"
                    )
                )
            self.name = 'full'
        else:
            user_reldate = 365*year
            search_results = Entrez.read(
                Entrez.esearch(
                    db=self.user_db, term=self.user_term, reldate=user_reldate, datetype="pdat", usehistory="y"
                    #db=self.user_db, term=user_term, datetype="pdat", usehistory="y"
                    )
                )
            self.name = str(year)

        count = int(search_results["Count"]) 
        return search_results, count
    
    def search_full(self, ix, data_dir, search_results, starting, count, batch): 
        batch_size = batch
        out_handle = open(os.path.join(data_dir, self.user_db+'.'+self.user_term+"."+str(ix)+"."+self.name+".txt"), "w") 
        for start in range(starting, count, batch_size):
            end = min(count, start + batch_size) 
            if end == count:
                batch=end-start 
            print("Going to download records from %i to %i" % (start + 1, end))
            fetch_handle = Entrez.efetch(
                db="pubmed",
                rettype="medline",
                retmode="text",
                retstart=start,
                retmax=batch_size,
                webenv=search_results["WebEnv"],
                query_key=search_results["QueryKey"],
                )
            data = fetch_handle.read()
            fetch_handle.close()
            out_handle.write(data)
        out_handle.close()
        
    def fetch_rec(self, rec_id, entrez_handle):
        fetch_handle = Entrez.efetch(db='pubmed', id=rec_id,
                                 rettype='Medline', retmode='text',
                                 webenv=entrez_handle['WebEnv'],
                                 query_key=entrez_handle['QueryKey'])
        rec = fetch_handle.read()
        return rec

