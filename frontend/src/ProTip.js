import * as React from 'react';
import Link from '@mui/material/Link';
import ArrowForwardIosIcon from '@mui/icons-material/ArrowForwardIos';
import Typography from '@mui/material/Typography';
import TextField from '@mui/material/TextField'
import Grid from '@mui/material/Grid'
import IconButton from '@mui/material/IconButton';
import {getProdText} from './handlers'

export default function ProTip() {

  const [prompt, setPrompt] = React.useState("")
  const [result, setResult] = React.useState("")
  const submitHandler = async () => {
    setResult("Loading generated text...")
    const msg = await getProdText(prompt)
    if (msg) {
      setResult(msg)
    }
  }

  const handleChange = (event) => {
    setPrompt(event.target.value)
  }

  return (
    <Grid
      container
    >
      <Grid item xs={12}>
        <Typography sx={{ mt: 6, mb: 3 }} color="text.secondary">
          trying to reproduce novel-length text with character sequence learning.
          <br></br> <br></br>
          <a href="https://github.com/rishabhsamb/deranged-murakami">github</a>
        </Typography>
      </Grid>
      <Grid
        item
        xs={11}
        style={{ display: "flex", gap: "1rem", alignItems: "center" }}
      >
        <TextField 
          id="outlined-basic" 
          label="start typing..." 
          variant="outlined" 
          fullWidth
          value={prompt}
          onChange={handleChange}
        />
        <IconButton 
          aria-label="submit"
          onClick={submitHandler}
        >
          <ArrowForwardIosIcon/>
        </IconButton>
        </Grid> 
        <Grid item xs={12}>
        <Typography sx={{ mt: 6, mb: 3 }} color="text.primary">
          {result}
        </Typography>
        </Grid>
    </Grid> 
  );
}
